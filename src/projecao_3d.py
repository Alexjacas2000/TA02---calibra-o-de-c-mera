"""Módulo de Projeção de Pontos 3D para 2D com OpenCV.

Realiza a projeção de pontos tridimensionais no espaço da imagem 2D
utilizando cv2.projectPoints, matriz de calibração intrínseca e vetores extrínsecos.
Executa validação quantitativa contra medições reais (cantos subpixel do tabuleiro)
e gera visualização tridimensional de eixos ortogonais e cubo em Realidade Aumentada.
"""

import os
import sys
import argparse
import json
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Projeção de Pontos 3D para 2D")
    parser.add_argument("--calib_data", type=str,
                        default=os.path.join("TA02", "dados", "resultados", "calibracao_camera.npz"),
                        help="Arquivo .npz com parâmetros de calibração")
    parser.add_argument("--img_dir", type=str, default=os.path.join("TA02", "Imagens"),
                        help="Diretório com imagens")
    parser.add_argument("--target_img", type=str, default="Foto (1).jpg",
                        help="Nome da imagem base para projeção")
    parser.add_argument("--corners_x", type=int, default=7,
                        help="Número de cantos internos em X")
    parser.add_argument("--corners_y", type=int, default=5,
                        help="Número de cantos internos em Y")
    parser.add_argument("--square_size", type=float, default=1.0,
                        help="Tamanho do quadrado")
    parser.add_argument("--output_dir", type=str, default=os.path.join("TA02", "dados", "resultados"),
                        help="Diretório de saída para dados numéricos")
    parser.add_argument("--evidence_dir", type=str, default=os.path.join("TA02", "evidencias"),
                        help="Diretório de saída para imagens de evidência")
    return parser.parse_args()


def draw_axes_and_cube(img, corners_proj, axis_proj, cube_proj):
    """Desenha eixos cartesianos e cubo 3D sobre a imagem."""
    out = img.copy()

    # Origem no primeiro canto
    origin = tuple(np.round(axis_proj[0].ravel()).astype(int))
    pt_x = tuple(np.round(axis_proj[1].ravel()).astype(int))
    pt_y = tuple(np.round(axis_proj[2].ravel()).astype(int))
    pt_z = tuple(np.round(axis_proj[3].ravel()).astype(int))

    # Eixos: X (Vermelho), Y (Verde), Z (Azul)
    cv2.line(out, origin, pt_x, (0, 0, 255), 4, cv2.LINE_AA)  # X - Vermelho
    cv2.line(out, origin, pt_y, (0, 255, 0), 4, cv2.LINE_AA)  # Y - Verde
    cv2.line(out, origin, pt_z, (255, 0, 0), 4, cv2.LINE_AA)  # Z - Azul

    cv2.putText(out, "X", pt_x, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(out, "Y", pt_y, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(out, "Z", pt_z, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

    # Cubo 3D: base e topo
    cube_pts = np.round(cube_proj.reshape(-1, 2)).astype(int)
    base = cube_pts[:4]
    top = cube_pts[4:]

    # Desenhar pilares verticais
    for i in range(4):
        cv2.line(out, tuple(base[i]), tuple(top[i]), (0, 215, 255), 2, cv2.LINE_AA)

    # Desenhar base e topo
    cv2.drawContours(out, [base], -1, (0, 165, 255), 2, cv2.LINE_AA)
    cv2.drawContours(out, [top], -1, (0, 255, 255), 3, cv2.LINE_AA)

    # Destacar cantos projetados reais do tabuleiro
    for pt in corners_proj.reshape(-1, 2):
        p = tuple(np.round(pt).astype(int))
        cv2.circle(out, p, 3, (255, 255, 0), -1, cv2.LINE_AA)

    return out


def run_projection(calib_data_path, img_dir, target_img_name, pattern_size, square_size, output_dir, evidence_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(evidence_dir, exist_ok=True)

    data = np.load(calib_data_path)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]
    rvecs = data["rvecs"]
    tvecs = data["tvecs"]
    valid_images = list(data["valid_images"])

    if target_img_name not in valid_images:
        target_idx = 0
        target_img_name = valid_images[0]
    else:
        target_idx = valid_images.index(target_img_name)

    rvec = rvecs[target_idx]
    tvec = tvecs[target_idx]

    img_path = os.path.join(img_dir, target_img_name)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Imagem não encontrada: {img_path}")

    # 1. Definir pontos 3D dos cantos do tabuleiro (Z = 0) para validação contra observações reais
    objp_corners = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp_corners[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

    # Detectar cantos reais na imagem alvo para comparação
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners_real = cv2.findChessboardCorners(gray, pattern_size, None)
    if not ret:
        raise RuntimeError("Não foi possível detectar cantos na imagem alvo para validação.")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_real_subpix = cv2.cornerSubPix(gray, corners_real, (11, 11), (-1, -1), criteria)

    # 2. Projetar os cantos 3D no plano 2D usando os parâmetros intrínsecos e extrínsecos
    corners_projected, _ = cv2.projectPoints(objp_corners, rvec, tvec, camera_matrix, dist_coeffs)

    # 3. Validação quantitativa real: erro euclidiano entre pontos reais medidos e projetados
    real_pts = corners_real_subpix.reshape(-1, 2)
    proj_pts = corners_projected.reshape(-1, 2)
    errors_per_point = np.linalg.norm(real_pts - proj_pts, axis=1)

    mean_err = float(np.mean(errors_per_point))
    std_err = float(np.std(errors_per_point))
    max_err = float(np.max(errors_per_point))
    min_err = float(np.min(errors_per_point))

    print(f"Validação da Projeção 3D->2D na imagem {target_img_name}:")
    print(f"  Pontos avaliados: {len(errors_per_point)}")
    print(f"  Erro médio euclidiano: {mean_err:.4f} pixels")
    print(f"  Desvio padrão do erro: {std_err:.4f} pixels")
    print(f"  Erro máximo: {max_err:.4f} pixels | Erro mínimo: {min_err:.4f} pixels")

    # 4. Definir pontos 3D para realidade aumentada: Eixos cartesianos e Cubo 3D
    # Eixos partem de (0,0,0) com comprimento de 3 quadrados
    axis_length = 3.0 * square_size
    axis_3d = np.float32([
        [0, 0, 0],
        [axis_length, 0, 0],
        [0, axis_length, 0],
        [0, 0, -axis_length]  # -Z sai do plano do tabuleiro em direção à câmera
    ])
    axis_proj, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)

    # Cubo de lado 2 sobre os cantos (0,0) a (2,2)
    s = 2.0 * square_size
    cube_3d = np.float32([
        [0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0],           # base no plano Z=0
        [0, 0, -s], [s, 0, -s], [s, s, -s], [0, s, -s]        # topo no plano Z=-s
    ])
    cube_proj, _ = cv2.projectPoints(cube_3d, rvec, tvec, camera_matrix, dist_coeffs)

    # Desenhar projeção sobre a imagem
    img_with_projection = draw_axes_and_cube(img, corners_projected, axis_proj, cube_proj)

    # Salvar evidência principal de projeção 3D
    evidence_path = os.path.join(evidence_dir, "projecao_3d.png")
    cv2.imwrite(evidence_path, img_with_projection)
    print(f"Evidência de projeção 3D salva em: {evidence_path}")

    # Salvar relatório com distinção clara entre dados reais e sintéticos
    projection_results = {
        "target_image": target_img_name,
        "validation_method": "Comparação entre pontos 3D conhecidos do tabuleiro projetados via cv2.projectPoints e observações 2D reais subpixel detectadas pelo cv2.findChessboardCorners/cornerSubPix",
        "real_points_count": len(real_pts),
        "metrics": {
            "mean_euclidean_error_pixels": mean_err,
            "std_euclidean_error_pixels": std_err,
            "max_euclidean_error_pixels": max_err,
            "min_euclidean_error_pixels": min_err
        },
        "synthetic_elements_rendered": {
            "cartesian_axes": {
                "origin": [0.0, 0.0, 0.0],
                "x_axis_end": [float(axis_length), 0.0, 0.0],
                "y_axis_end": [0.0, float(axis_length), 0.0],
                "z_axis_end": [0.0, 0.0, float(-axis_length)]
            },
            "cube_3d": {
                "dimensions": [float(s), float(s), float(s)],
                "vertices_count": 8
            }
        },
        "camera_extrinsic_vectors": {
            "rotation_vector_rvec": rvec.tolist(),
            "translation_vector_tvec": tvec.tolist()
        },
        "evidence_files": [
            "projecao_3d.png"
        ]
    }

    report_path = os.path.join(output_dir, "projecao_3d.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(projection_results, f, indent=2, ensure_ascii=False)
    print(f"Relatório de projeção 3D salvo em: {report_path}")

    return projection_results


if __name__ == "__main__":
    args = parse_args()
    run_projection(
        calib_data_path=args.calib_data,
        img_dir=args.img_dir,
        target_img_name=args.target_img,
        pattern_size=(args.corners_x, args.corners_y),
        square_size=args.square_size,
        output_dir=args.output_dir,
        evidence_dir=args.evidence_dir
    )
