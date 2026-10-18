"""Módulo de Calibração de Câmera com OpenCV.

Executa a detecção dos cantos do tabuleiro de xadrez em fotografias,
refina as posições subpixel, calcula os parâmetros intrínsecos e extrínsecos,
avalia o erro de reprojeção global e por imagem, e exporta os resultados
e evidências visuais.
"""

import os
import sys
import argparse
import json
import re
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def natural_sort_key(s):
    """Ordenação natural baseada no número dentro dos parênteses."""
    match = re.search(r'\((\d+)\)', s)
    if match:
        return int(match.group(1))
    return s


def parse_args():
    parser = argparse.ArgumentParser(description="Calibração de Câmera usando OpenCV")
    parser.add_argument("--img_dir", type=str, default=os.path.join("TA02", "Imagens"),
                        help="Diretório contendo as imagens de calibração")
    parser.add_argument("--corners_x", type=int, default=7,
                        help="Número de cantos internos na horizontal")
    parser.add_argument("--corners_y", type=int, default=5,
                        help="Número de cantos internos na vertical")
    parser.add_argument("--square_size", type=float, default=1.0,
                        help="Tamanho do lado do quadrado do tabuleiro (1.0 = unidades de grade)")
    parser.add_argument("--square_unit", type=str, default="unidades",
                        help="Unidade de medida do quadrado (ex: 'unidades', 'mm')")
    parser.add_argument("--output_dir", type=str, default=os.path.join("TA02", "dados", "resultados"),
                        help="Diretório para salvar os resultados numéricos")
    parser.add_argument("--evidence_dir", type=str, default=os.path.join("TA02", "evidencias"),
                        help="Diretório para salvar as imagens de evidência")
    return parser.parse_args()


def calibrate(img_dir, pattern_size, square_size, square_unit, output_dir, evidence_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(evidence_dir, exist_ok=True)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Preparar pontos no espaço do objeto 3D (Z = 0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

    objpoints = []  # pontos 3D no espaço do mundo
    imgpoints = []  # pontos 2D no plano da imagem

    image_files = sorted(
        [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))],
        key=natural_sort_key
    )
    if not image_files:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {img_dir}")

    print(f"Total de imagens encontradas: {len(image_files)}")
    print(f"Padrão de cantos internos: {pattern_size[0]} x {pattern_size[1]}")
    print(f"Escala do quadrado: {square_size} {square_unit}")

    valid_images = []
    rejected_images = []
    img_shape = None
    drawn_evidence_saved = False

    for fname in image_files:
        fpath = os.path.join(img_dir, fname)
        img = cv2.imread(fpath)
        if img is None:
            rejected_images.append({"file": fname, "reason": "Falha ao carregar imagem"})
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]  # (largura, altura)

        # Encontrar cantos do tabuleiro
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            # Refinamento subpixel
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            valid_images.append(fname)

            # Salvar primeira imagem como evidência principal do tabuleiro detectado
            if not drawn_evidence_saved:
                img_drawn = img.copy()
                cv2.drawChessboardCorners(img_drawn, pattern_size, corners2, ret)
                evidence_path = os.path.join(evidence_dir, "tabuleiro_detectado.png")
                cv2.imwrite(evidence_path, img_drawn)
                print(f"Evidência de detecção salva em: {evidence_path}")
                drawn_evidence_saved = True
        else:
            rejected_images.append({"file": fname, "reason": "Cantos do tabuleiro não detectados"})

    print(f"\nImagens válidas para calibração: {len(valid_images)}/{len(image_files)}")
    if rejected_images:
        print(f"Imagens rejeitadas: {len(rejected_images)}")
        for rej in rejected_images:
            print(f"  - {rej['file']}: {rej['reason']}")

    if len(valid_images) < 3:
        raise ValueError("Quantidade insuficiente de imagens válidas para calibração (mínimo recomendado: 3).")

    # Executar Calibração da Câmera
    print("\nExecutando cv2.calibrateCamera...")
    ret_rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )

    # Calcular Erro de Reprojeção detalhado por imagem
    per_image_errors = []
    total_error = 0
    total_points = 0

    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        pts_orig = imgpoints[i].reshape(-1, 2)
        pts_proj = imgpoints2.reshape(-1, 2)
        diff = pts_orig - pts_proj
        # Erro RMS por ponto na imagem i
        error = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
        per_image_errors.append({
            "image": valid_images[i],
            "rmse_pixels": error
        })
        total_error += error ** 2 * len(pts_proj)
        total_points += len(pts_proj)

    mean_reprojection_error = float(np.sqrt(total_error / total_points))

    # Formatar resultados
    calibration_results = {
        "pattern_size": list(pattern_size),
        "square_size": square_size,
        "square_unit": square_unit,
        "image_size": {
            "width": int(img_shape[0]),
            "height": int(img_shape[1])
        },
        "valid_images_count": len(valid_images),
        "rejected_images_count": len(rejected_images),
        "valid_images": valid_images,
        "rejected_images": rejected_images,
        "rms_opencv": float(ret_rms),
        "mean_reprojection_error_pixels": mean_reprojection_error,
        "camera_matrix": {
            "fx": float(mtx[0, 0]),
            "fy": float(mtx[1, 1]),
            "cx": float(mtx[0, 2]),
            "cy": float(mtx[1, 2]),
            "full_matrix": mtx.tolist()
        },
        "distortion_coefficients": {
            "k1": float(dist[0, 0]),
            "k2": float(dist[0, 1]),
            "p1": float(dist[0, 2]),
            "p2": float(dist[0, 3]),
            "k3": float(dist[0, 4]) if dist.shape[1] > 4 else 0.0,
            "full_vector": dist.tolist()[0]
        },
        "per_image_errors": per_image_errors
    }

    # Salvar em JSON
    json_path = os.path.join(output_dir, "calibracao_camera.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(calibration_results, f, indent=2, ensure_ascii=False)
    print(f"\nResultados salvos em JSON: {json_path}")

    # Salvar em NPZ para consumo dos outros scripts
    npz_path = os.path.join(output_dir, "calibracao_camera.npz")
    np.savez(
        npz_path,
        camera_matrix=mtx,
        dist_coeffs=dist,
        rvecs=np.array(rvecs),
        tvecs=np.array(tvecs),
        valid_images=np.array(valid_images),
        image_shape=np.array(img_shape)
    )
    print(f"Matrizes salvas em formato NumPy binário: {npz_path}")

    # Gerar Gráfico de Erro de Reprojeção por Imagem com ordenação natural
    plt.figure(figsize=(11, 5.5))
    names = [f"Foto {re.search(r'\((\d+)\)', item['image']).group(1)}" for item in per_image_errors]
    errors = [item["rmse_pixels"] for item in per_image_errors]
    bars = plt.bar(names, errors, color="#1f77b4", edgecolor="#0b3b60", alpha=0.85)

    # Inserir rótulo de valor numérico sobre cada barra
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h + 0.05, f"{h:.2f}",
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1c3d5a")

    plt.axhline(mean_reprojection_error, color="red", linestyle="--", linewidth=1.5,
                label=f"Média Global: {mean_reprojection_error:.4f} px")
    plt.xlabel("Fotografia de Calibração", fontsize=11, fontweight="bold")
    plt.ylabel("Erro RMS de Reprojeção (pixels)", fontsize=11, fontweight="bold")
    plt.title("Erro de Reprojeção Individual por Fotografia", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9.5)
    plt.ylim(0, max(errors) * 1.15)
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.legend(fontsize=10.5, loc="upper right")
    plt.tight_layout()

    error_plot_path = os.path.join(evidence_dir, "erro_reprojecao.png")
    plt.savefig(error_plot_path, dpi=300)
    plt.close()
    print(f"Gráfico de erro salvo em: {error_plot_path}")

    print("\n--- RESUMO DA CALIBRAÇÃO ---")
    print(f"Resolução da Imagem: {img_shape[0]} x {img_shape[1]} pixels")
    print(f"Distância Focal: fx = {mtx[0,0]:.2f} px, fy = {mtx[1,1]:.2f} px")
    print(f"Ponto Principal: cx = {mtx[0,2]:.2f} px, cy = {mtx[1,2]:.2f} px")
    print(f"Coeficientes de Distorção: k1={dist[0,0]:.5f}, k2={dist[0,1]:.5f}, p1={dist[0,2]:.5f}, p2={dist[0,3]:.5f}, k3={dist[0,4]:.5f}")
    print(f"Erro Médio de Reprojeção: {mean_reprojection_error:.4f} pixels")
    print("----------------------------\n")

    return calibration_results


if __name__ == "__main__":
    args = parse_args()
    calibrate(
        img_dir=args.img_dir,
        pattern_size=(args.corners_x, args.corners_y),
        square_size=args.square_size,
        square_unit=args.square_unit,
        output_dir=args.output_dir,
        evidence_dir=args.evidence_dir
    )
