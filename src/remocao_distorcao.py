"""Módulo de Remoção de Distorção de Imagens com OpenCV.

Carrega os parâmetros intrínsecos e coeficientes de distorção obtidos
pela calibração da câmera e aplica a retificação/desdistorção utilizando
cv2.undistort, cv2.getOptimalNewCameraMatrix e cv2.initUndistortRectifyMap + cv2.remap.
Gera comparações visuais e analisa impacto no campo de visão (FOV).
"""

import os
import sys
import argparse
import json
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Remoção de Distorção de Imagens")
    parser.add_argument("--calib_data", type=str,
                        default=os.path.join("TA02", "dados", "resultados", "calibracao_camera.npz"),
                        help="Arquivo .npz com parâmetros de calibração")
    parser.add_argument("--img_dir", type=str, default=os.path.join("TA02", "Imagens"),
                        help="Diretório com imagens originais")
    parser.add_argument("--target_img", type=str, default="Foto (1).jpg",
                        help="Nome da imagem a ser processada e comparada")
    parser.add_argument("--output_dir", type=str, default=os.path.join("TA02", "dados", "resultados"),
                        help="Diretório para salvar métricas em JSON")
    parser.add_argument("--evidence_dir", type=str, default=os.path.join("TA02", "evidencias"),
                        help="Diretório para salvar evidências visuais")
    return parser.parse_args()


def remove_distortion(calib_data_path, img_dir, target_img_name, output_dir, evidence_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(evidence_dir, exist_ok=True)

    if not os.path.exists(calib_data_path):
        raise FileNotFoundError(f"Arquivo de calibração não encontrado: {calib_data_path}")

    data = np.load(calib_data_path)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]
    img_shape = tuple(data["image_shape"])  # (w, h)

    target_path = os.path.join(img_dir, target_img_name)
    img = cv2.imread(target_path)
    if img is None:
        raise FileNotFoundError(f"Imagem alvo não encontrada: {target_path}")

    h, w = img.shape[:2]
    print(f"Processando imagem: {target_img_name} ({w}x{h})")

    # 1. Matrizes ótimas com alpha=0 (todos pixels pretos cortados) e alpha=1 (todos pixels originais mantidos)
    newcameramtx_alpha0, roi_alpha0 = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 0, (w, h)
    )
    newcameramtx_alpha1, roi_alpha1 = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )

    # 2. Método 1: cv2.undistort direto
    dst_undistort_alpha0 = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx_alpha0)
    dst_undistort_alpha1 = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx_alpha1)

    # 3. Método 2: cv2.initUndistortRectifyMap + cv2.remap
    mapx, mapy = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, newcameramtx_alpha0, (w, h), 5
    )
    dst_remap_alpha0 = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # Verificação numérica de equivalência entre os métodos
    diff_methods = np.mean(np.abs(dst_undistort_alpha0.astype(float) - dst_remap_alpha0.astype(float)))
    print(f"Diferença média entre cv2.undistort e cv2.remap: {diff_methods:.6f} (valores próximos de 0 confirmam equivalência)")

    # Salvar evidências individuais
    cv2.imwrite(os.path.join(evidence_dir, "imagem_original.png"), img)
    cv2.imwrite(os.path.join(evidence_dir, "imagem_corrigida.png"), dst_undistort_alpha0)
    cv2.imwrite(os.path.join(evidence_dir, "imagem_corrigida_alpha1.png"), dst_undistort_alpha1)

    # Gerar imagem comparativa com linhas-guia para demonstrar a retificação das linhas do tabuleiro
    comp_orig = img.copy()
    comp_corr = dst_undistort_alpha0.copy()

    # Traçar linhas de referência horizontais e verticais para evidenciar retilineidade
    grid_color = (0, 0, 255)  # Vermelho
    for y in range(120, h - 80, 120):
        cv2.line(comp_orig, (0, y), (w, y), grid_color, 1, cv2.LINE_AA)
        cv2.line(comp_corr, (0, y), (w, y), grid_color, 1, cv2.LINE_AA)
    for x in range(160, w - 100, 160):
        cv2.line(comp_orig, (x, 0), (x, h), grid_color, 1, cv2.LINE_AA)
        cv2.line(comp_corr, (x, 0), (x, h), grid_color, 1, cv2.LINE_AA)

    # Legendas sobre as imagens
    cv2.putText(comp_orig, "ORIGINAL (com distorcao)", (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(comp_corr, "CORRIGIDA (undistort alpha=0)", (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    comparison_side_by_side = np.hstack([comp_orig, comp_corr])
    comparison_path = os.path.join(evidence_dir, "comparacao_distorcao.png")
    cv2.imwrite(comparison_path, comparison_side_by_side)
    print(f"Comparativo lado a lado salvo em: {comparison_path}")

    # Salvar métricas e análise no JSON
    distortion_report = {
        "target_image": target_img_name,
        "image_dimensions": {"width": w, "height": h},
        "methods_evaluated": [
            "cv2.undistort",
            "cv2.initUndistortRectifyMap + cv2.remap"
        ],
        "methods_mean_absolute_difference": float(diff_methods),
        "optimal_camera_matrix_alpha0": newcameramtx_alpha0.tolist(),
        "optimal_camera_matrix_alpha1": newcameramtx_alpha1.tolist(),
        "roi_alpha0": [int(x) for x in roi_alpha0],
        "roi_alpha1": [int(x) for x in roi_alpha1],
        "analysis": {
            "alpha_0": "Remove completamente as bordas pretas introduzidas pela desdistorção, com pequeno corte no campo de visão periférico.",
            "alpha_1": "Preserva todos os pixels do sensor original, gerando bordas curvas pretas nos cantos onde a distorção em almofada/barril foi descomprimida.",
            "distortion_types_present": "Distorção radial moderada (k1, k2, k3) com leve distorção tangencial (p1, p2)."
        },
        "evidence_files": [
            "imagem_original.png",
            "imagem_corrigida.png",
            "imagem_corrigida_alpha1.png",
            "comparacao_distorcao.png"
        ]
    }

    report_path = os.path.join(output_dir, "remocao_distorcao.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(distortion_report, f, indent=2, ensure_ascii=False)
    print(f"Relatório de desdistorção salvo em: {report_path}")

    return distortion_report


if __name__ == "__main__":
    args = parse_args()
    remove_distortion(
        calib_data_path=args.calib_data,
        img_dir=args.img_dir,
        target_img_name=args.target_img,
        output_dir=args.output_dir,
        evidence_dir=args.evidence_dir
    )
