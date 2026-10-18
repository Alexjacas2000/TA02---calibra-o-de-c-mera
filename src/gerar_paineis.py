"""Módulo para geração de painéis consolidados em imagens únicas (.png).

Gera compilações visuais completas com todas as 16 imagens processadas:
1. Painel de 16 imagens de detecção de cantos (4 colunas x 4 linhas);
2. Painel de 16 imagens de projeção 3D / Realidade Aumentada (4 colunas x 4 linhas);
3. Painel comparativo de desdistorção (2 colunas de pares x 8 linhas).
"""

import os
import re
import cv2
import numpy as np


def natural_sort_key(s):
    """Ordenação natural para 'Foto (1).jpg', 'Foto (2).jpg', ..., 'Foto (16).jpg'."""
    match = re.search(r'\((\d+)\)', s)
    if match:
        return int(match.group(1))
    return s


def build_mosaic_4col_4rows(images, titles, thumb_w=480, thumb_h=270):
    """Gera um mosaico de 4 colunas por 4 linhas a partir de 16 imagens."""
    assert len(images) == 16, f"Esperadas 16 imagens, recebidas {len(images)}"

    rows = []
    for r in range(4):
        row_imgs = []
        for c in range(4):
            idx = r * 4 + c
            img = images[idx]
            title = titles[idx]

            resized = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

            # Banner superior para cada miniatura
            banner = np.zeros((32, thumb_w, 3), dtype=np.uint8)
            banner[:] = (35, 35, 35)
            cv2.putText(banner, title, (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            cell = np.vstack([banner, resized])
            cell = cv2.copyMakeBorder(cell, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[70, 70, 70])
            row_imgs.append(cell)

        row_concat = np.hstack(row_imgs)
        rows.append(row_concat)

    mosaic = np.vstack(rows)
    return mosaic


def build_mosaic_distortion_2col_8rows(pairs, titles, thumb_w=320, thumb_h=180):
    """Gera um mosaico de 2 colunas de pares [Orig | Corr] por 8 linhas (16 pares ao todo)."""
    assert len(pairs) == 16, f"Esperados 16 pares, recebidos {len(pairs)}"

    pair_w = thumb_w * 2  # largura de um par (Original + Corrigida)
    rows = []

    for r in range(8):
        row_cells = []
        for c in range(2):
            idx = r * 2 + c
            orig_img, corr_img = pairs[idx]
            title = titles[idx]

            r_orig = cv2.resize(orig_img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
            r_corr = cv2.resize(corr_img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

            # Banner do par
            banner = np.zeros((28, pair_w, 3), dtype=np.uint8)
            banner[:] = (40, 40, 40)
            cv2.putText(banner, f"{title}: Orig (esq) vs Corr (dir)", (10, 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            pair_img = np.hstack([r_orig, r_corr])
            cell = np.vstack([banner, pair_img])
            cell = cv2.copyMakeBorder(cell, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[80, 80, 80])
            row_cells.append(cell)

        row_concat = np.hstack(row_cells)
        rows.append(row_concat)

    mosaic = np.vstack(rows)
    return mosaic


def draw_axes_and_cube_on_img(img, rvec, tvec, camera_matrix, dist_coeffs, pattern_size, square_size=1.0):
    """Projeta cantos, eixos XYZ e cubo 3D sobre a imagem."""
    out = img.copy()

    # 1. Cantos do tabuleiro
    objp_corners = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp_corners[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
    corners_proj, _ = cv2.projectPoints(objp_corners, rvec, tvec, camera_matrix, dist_coeffs)

    for pt in corners_proj.reshape(-1, 2):
        p = tuple(np.round(pt).astype(int))
        cv2.circle(out, p, 3, (255, 255, 0), -1, cv2.LINE_AA)

    # 2. Eixos cartesianos
    axis_len = 3.0 * square_size
    axis_3d = np.float32([
        [0, 0, 0],
        [axis_len, 0, 0],
        [0, axis_len, 0],
        [0, 0, -axis_len]
    ])
    axis_proj, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)

    origin = tuple(np.round(axis_proj[0].ravel()).astype(int))
    pt_x = tuple(np.round(axis_proj[1].ravel()).astype(int))
    pt_y = tuple(np.round(axis_proj[2].ravel()).astype(int))
    pt_z = tuple(np.round(axis_proj[3].ravel()).astype(int))

    cv2.line(out, origin, pt_x, (0, 0, 255), 3, cv2.LINE_AA)  # X Vermelho
    cv2.line(out, origin, pt_y, (0, 255, 0), 3, cv2.LINE_AA)  # Y Verde
    cv2.line(out, origin, pt_z, (255, 0, 0), 3, cv2.LINE_AA)  # Z Azul

    # 3. Cubo 3D
    s = 2.0 * square_size
    cube_3d = np.float32([
        [0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0],
        [0, 0, -s], [s, 0, -s], [s, s, -s], [0, s, -s]
    ])
    cube_proj, _ = cv2.projectPoints(cube_3d, rvec, tvec, camera_matrix, dist_coeffs)
    cube_pts = np.round(cube_proj.reshape(-1, 2)).astype(int)
    base = cube_pts[:4]
    top = cube_pts[4:]

    for i in range(4):
        cv2.line(out, tuple(base[i]), tuple(top[i]), (0, 215, 255), 2, cv2.LINE_AA)
    cv2.drawContours(out, [base], -1, (0, 165, 255), 2, cv2.LINE_AA)
    cv2.drawContours(out, [top], -1, (0, 255, 255), 2, cv2.LINE_AA)

    return out


def generate_all_panels():
    img_dir = os.path.join("TA02", "Imagens")
    calib_npz_path = os.path.join("TA02", "dados", "resultados", "calibracao_camera.npz")
    evidence_dir = os.path.join("TA02", "evidencias")
    latex_fig_dir = os.path.join("TA02 Latex", "figuras")

    os.makedirs(evidence_dir, exist_ok=True)
    os.makedirs(latex_fig_dir, exist_ok=True)

    data = np.load(calib_npz_path)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]
    rvecs = data["rvecs"]
    tvecs = data["tvecs"]
    valid_images = list(data["valid_images"])

    extrinsic_map = {valid_images[i]: (rvecs[i], tvecs[i]) for i in range(len(valid_images))}

    all_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
                       key=natural_sort_key)
    print(f"Total de fotos ordenadas naturalmente: {len(all_files)}")

    pattern_size = (7, 5)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    detection_imgs = []
    projection_imgs = []
    distortion_pairs = []
    titles = []

    w_ref, h_ref = 1280, 720
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w_ref, h_ref), 0, (w_ref, h_ref))

    for fname in all_files:
        path = os.path.join(img_dir, fname)
        img = cv2.imread(path)
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        titles.append(fname.replace(".jpg", ""))

        # 1. Detecção de cantos
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_det = img.copy()
        cv2.drawChessboardCorners(img_det, pattern_size, corners2, ret)
        detection_imgs.append(img_det)

        # 2. Projeção 3D
        rvec, tvec = extrinsic_map[fname]
        img_proj = draw_axes_and_cube_on_img(img, rvec, tvec, camera_matrix, dist_coeffs, pattern_size, 1.0)
        projection_imgs.append(img_proj)

        # 3. Par para comparação de distorção
        undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)
        comp_orig = img.copy()
        comp_undist = undistorted.copy()
        grid_col = (0, 0, 255)
        for y in range(120, h - 60, 120):
            cv2.line(comp_orig, (0, y), (w, y), grid_col, 1, cv2.LINE_AA)
            cv2.line(comp_undist, (0, y), (w, y), grid_col, 1, cv2.LINE_AA)
        distortion_pairs.append((comp_orig, comp_undist))

    # Mosaico 1: 16 imagens de detecção (4 colunas x 4 linhas)
    print("\nConstruindo painel 1: 16 imagens de detecção de cantos (4 colunas x 4 linhas)...")
    mosaic_det = build_mosaic_4col_4rows(detection_imgs, titles, thumb_w=480, thumb_h=270)
    out_det_path = os.path.join(evidence_dir, "painel_16_deteccoes.png")
    cv2.imwrite(out_det_path, mosaic_det)
    cv2.imwrite(os.path.join(latex_fig_dir, "painel_16_deteccoes.png"), mosaic_det)
    print(f"Salvo: {out_det_path} ({mosaic_det.shape[1]}x{mosaic_det.shape[0]} px)")

    # Mosaico 2: 16 imagens de projeção 3D (4 colunas x 4 linhas)
    print("\nConstruindo painel 2: 16 imagens de projeção 3D (4 colunas x 4 linhas)...")
    mosaic_proj = build_mosaic_4col_4rows(projection_imgs, titles, thumb_w=480, thumb_h=270)
    out_proj_path = os.path.join(evidence_dir, "painel_16_projecoes_3d.png")
    cv2.imwrite(out_proj_path, mosaic_proj)
    cv2.imwrite(os.path.join(latex_fig_dir, "painel_16_projecoes_3d.png"), mosaic_proj)
    print(f"Salvo: {out_proj_path} ({mosaic_proj.shape[1]}x{mosaic_proj.shape[0]} px)")

    # Mosaico 3: Comparação de distorção das 16 fotos (2 colunas de pares x 8 linhas)
    print("\nConstruindo painel 3: Comparação de distorção das 16 fotos (2 colunas de pares x 8 linhas)...")
    mosaic_dist = build_mosaic_distortion_2col_8rows(distortion_pairs, titles, thumb_w=320, thumb_h=180)
    out_dist_path = os.path.join(evidence_dir, "painel_16_comparacoes_distorcao.png")
    cv2.imwrite(out_dist_path, mosaic_dist)
    cv2.imwrite(os.path.join(latex_fig_dir, "painel_16_comparacoes_distorcao.png"), mosaic_dist)
    print(f"Salvo: {out_dist_path} ({mosaic_dist.shape[1]}x{mosaic_dist.shape[0]} px)")

    print("\nTodos os novos painéis (4 colunas para detecção/projeção e 2 colunas para distorção) foram gerados com sucesso!")


if __name__ == "__main__":
    generate_all_panels()
