import hashlib
import json
import os
import re

import bpy
import numpy as np
from PIL import Image

from step0_mat_map import classify_material_role, extract_material_base_image

_SUPPORTED_EXTS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".tga",
    ".bmp",
    ".webp",
    ".dds",
    ".tif",
    ".tiff",
)
_SKIP_TEX_DIRS = {"vrm_temp_baked", "spa", "sph", "toon", "toons", "__pycache__"}
_EXTRA_TEX_NAMES = {
    "tex",
    "texture",
    "textures",
    "texture2d",
    "img",
    "images",
    "map",
    "maps",
}


def _clean_path_basename(path_or_name: str) -> str:
    norm = (path_or_name or "").replace("\\", "/")
    return os.path.basename(norm)


def normalize_texture_dirs(texture_dir_or_dirs):
    """兼容 str / ;分隔多路径 / list，并自动补全 .blend 周围及已引用贴图所在目录"""
    if isinstance(texture_dir_or_dirs, (list, tuple, set)):
        raw_list = [str(x).strip() for x in texture_dir_or_dirs if x and str(x).strip()]
    elif texture_dir_or_dirs:
        raw_list = [x.strip() for x in str(texture_dir_or_dirs).split(";") if x.strip()]
    else:
        raw_list = []

    blend_dir = bpy.path.abspath("//")
    candidates = []
    for raw in raw_list:
        p = (
            raw
            if os.path.isabs(raw)
            else os.path.abspath(os.path.join(blend_dir or ".", raw))
        )
        candidates.append(p)

    probe_roots = []
    if blend_dir and os.path.isdir(blend_dir):
        probe_roots.append(blend_dir)
    for p in list(candidates):
        parent = os.path.dirname(os.path.abspath(p))
        if parent and os.path.isdir(parent):
            probe_roots.append(parent)

    for root in probe_roots:
        try:
            for entry in sorted(os.listdir(root)):
                low = entry.lower()
                if low in _SKIP_TEX_DIRS:
                    continue
                full = os.path.join(root, entry)
                if os.path.isdir(full) and (
                    low in _EXTRA_TEX_NAMES or low.startswith(("tex", "texture"))
                ):
                    candidates.append(full)
        except OSError:
            pass

    if blend_dir and os.path.isdir(blend_dir):
        candidates.append(blend_dir)

    for img in bpy.data.images:
        fp = (img.filepath or "").replace("\\", "/")
        if not fp:
            continue
        abs_fp = bpy.path.abspath(fp)
        d = os.path.dirname(abs_fp)
        if d and os.path.isdir(d) and os.path.basename(d).lower() not in _SKIP_TEX_DIRS:
            candidates.append(d)

    unique_dirs = []
    seen = set()
    for d in candidates:
        abs_d = os.path.abspath(d)
        norm = os.path.normcase(abs_d)
        if norm not in seen and os.path.isdir(abs_d):
            seen.add(norm)
            unique_dirs.append(abs_d)
    return unique_dirs


def build_texture_index(texture_dirs):
    """跨多个贴图目录建立文件名与词干索引（先入为主，跳过 spa/toon/vrm_temp_baked）"""
    if isinstance(texture_dirs, str):
        texture_dirs = [texture_dirs]

    blend_dir_norm = os.path.normcase(os.path.abspath(bpy.path.abspath("//") or ""))
    by_filename = {}
    by_stem = {}

    for base_dir in texture_dirs:
        if not base_dir or not os.path.exists(base_dir):
            continue
        is_root_blend = os.path.normcase(os.path.abspath(base_dir)) == blend_dir_norm
        for root, dirs, files in os.walk(base_dir):
            dirs[:] = [d for d in dirs if d.lower() not in _SKIP_TEX_DIRS]
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in _SUPPORTED_EXTS:
                    continue
                full_p = os.path.join(root, fname)
                by_filename.setdefault(fname.lower(), full_p)
                stem = os.path.splitext(fname)[0].lower()
                by_stem.setdefault(stem, full_p)
            if is_root_blend:
                break

    return by_filename, by_stem


def resolve_disk_path(img, texture_dirs, tex_index=None):
    if not img:
        return None
    if isinstance(texture_dirs, str):
        texture_dirs = [texture_dirs]

    blend_dir = bpy.path.abspath("//")
    raw_fp = (img.filepath or "").replace("\\", "/")
    if raw_fp:
        abs_p = bpy.path.abspath(raw_fp)
        if os.path.exists(abs_p) and os.path.isfile(abs_p):
            return abs_p
        if raw_fp.startswith("//") and blend_dir:
            cand_rel = os.path.abspath(os.path.join(blend_dir, raw_fp[2:]))
            if os.path.exists(cand_rel) and os.path.isfile(cand_rel):
                return cand_rel

    by_filename, by_stem = tex_index or build_texture_index(texture_dirs)
    candidates = []
    if raw_fp:
        candidates.append(_clean_path_basename(raw_fp))
    candidates.append(re.sub(r"\.\d{3}$", "", _clean_path_basename(img.name)))

    for cand in candidates:
        if not cand:
            continue
        for td in texture_dirs:
            direct = os.path.join(td, cand)
            if os.path.exists(direct) and os.path.isfile(direct):
                return direct
        hit = by_filename.get(cand.lower())
        if hit:
            return hit

    for cand in candidates:
        stem = os.path.splitext(cand)[0].lower()
        if stem and stem in by_stem:
            return by_stem[stem]

    return None


def image_to_pil_rgba(img_obj, disk_path):
    if disk_path and os.path.exists(disk_path):
        return Image.open(disk_path).convert("RGBA")
    if img_obj and img_obj.has_data and img_obj.size[0] > 0 and img_obj.size[1] > 0:
        w, h = img_obj.size[0], img_obj.size[1]
        px = np.array(img_obj.pixels[:], dtype=np.float32).reshape((h, w, 4))
        px = np.flipud(px)
        return Image.fromarray(
            np.clip(px * 255.0, 0, 255).astype(np.uint8), mode="RGBA"
        )
    return None


def find_emissive_counterpart(disk_path, texture_dirs, tex_index=None):
    if not disk_path or not texture_dirs:
        return None
    if isinstance(texture_dirs, str):
        texture_dirs = [texture_dirs]

    _, by_stem = tex_index or build_texture_index(texture_dirs)
    stem = os.path.splitext(os.path.basename(disk_path))[0]

    candidates_prefix = []
    for suffix in (
        "_D_noalpha2",
        "_D_noalpha",
        "_D1",
        "_D2",
        "_D",
        "_DH",
        "_Dif",
        "_Diffuse",
        "_Albedo",
        "_BaseColor",
    ):
        if stem.lower().endswith(suffix.lower()):
            base_prefix = stem[: -len(suffix)]
            candidates_prefix.extend(
                [
                    f"{base_prefix}_E",
                    f"{base_prefix}_Emissive",
                    f"{base_prefix}_EMI",
                    f"{base_prefix}_Glow",
                    f"{base_prefix}_Illum",
                ]
            )
            break

    if not candidates_prefix:
        candidates_prefix.extend(
            [
                f"{stem}_E",
                f"{stem}_Emissive",
                f"{stem}_EMI",
                f"{stem}_Glow",
                f"{stem}_发光",
            ]
        )

    for cand in candidates_prefix:
        hit = by_stem.get(cand.lower())
        if hit and os.path.exists(hit):
            return hit
    return None


def is_plus_overlay_material(mat_name: str, img_name: str = "") -> bool:
    m = re.sub(r"\.\d{3}$", "", (mat_name or "").strip().lower())
    i = re.sub(r"\.\d{3}$", "", (img_name or "").strip().lower())
    return "+" in m or "+" in i or m.startswith("手2")


def get_alpha_policy(role: str, mat_name: str = "", img_name: str = "") -> str:
    if role == "fx_overlay":
        return "fx"
    if role in ("lash_brow", "cutout_cloth"):
        return "alpha"
    if is_plus_overlay_material(mat_name, img_name):
        return "plus"
    return "opaque"


def maybe_synthesize_black_bg_alpha(arr, role, is_plus=False):
    if role != "fx_overlay" and not is_plus:
        return arr
    alpha = arr[:, :, 3]
    if np.any(alpha < 250):
        return arr

    luma = np.max(arr[:, :, :3], axis=2)
    black_ratio = float(np.mean(luma < 18.0))
    threshold = 0.30 if role == "fx_overlay" else 0.55
    if black_ratio > threshold:
        new_alpha = np.clip((luma - 12.0) * (255.0 / 243.0), 0.0, 255.0)
        arr = arr.copy()
        arr[:, :, 3] = new_alpha
    return arr


def detect_alpha_mode(rgba_arr, role, mat_or_img_name, is_plus=False):
    name_low = mat_or_img_name.lower()
    if role == "fx_overlay":
        return True, "BLEND"

    alpha = rgba_arr[:, :, 3]
    transparent_ratio = float(np.mean(alpha < 20.0))
    non_opaque = alpha < 250
    has_non_opaque = bool(np.any(non_opaque))

    # 对 ZZZ 的 + 叠层材质/贴图（肌+、体+、发+、体2金属+、手2）：
    # 只要真实存在大面积透明或扣出来的透明像素，就按透明贴图保留 PNG，绝不压成黑底 JPG
    if is_plus:
        if transparent_ratio >= 0.05:
            semi = (alpha > 10) & (alpha < 245)
            semi_ratio = float(np.sum(semi)) / max(1, int(np.sum(non_opaque)))
            return True, ("BLEND" if semi_ratio > 0.35 else "MASK")
        return False, "OPAQUE"

    # 普通非叠层的脸/皮肤/眼睛/口腔/头发主体/金属/常规服装 强制 OPAQUE
    if role not in ("lash_brow", "cutout_cloth"):
        return False, "OPAQUE"

    if "noalpha" in name_low and "alpha" not in name_low.replace("noalpha", ""):
        return False, "OPAQUE"

    if not has_non_opaque:
        return False, "OPAQUE"

    semi = (alpha > 10) & (alpha < 245)
    semi_ratio = float(np.sum(semi)) / max(1, int(np.sum(non_opaque)))
    if semi_ratio > 0.35 and role != "lash_brow":
        return True, "BLEND"
    return True, "MASK"


def resize_if_needed(pil_img, max_size):
    if max_size <= 0:
        return pil_img
    w, h = pil_img.size
    if max(w, h) <= max_size:
        return pil_img
    scale = max_size / float(max(w, h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def run(texture_dir, max_tex_size=1024):
    tex_dirs = normalize_texture_dirs(texture_dir)
    print(
        f">>> [步骤 4] 优化 ZZZ TEX 贴图（搜索 {len(tex_dirs)} 个目录 | 全图上限 {max_tex_size}px + 不透明转JPEG）..."
    )
    if not tex_dirs:
        return

    primary_dir = tex_dirs[0]
    temp_dir = os.path.join(primary_dir, "vrm_temp_baked")
    os.makedirs(temp_dir, exist_ok=True)
    tex_index = build_texture_index(tex_dirs)

    image_jobs = {}
    for mat in bpy.data.materials:
        if not mat or getattr(mat, "users", 0) == 0:
            continue
        base_img = extract_material_base_image(mat)
        if not base_img:
            continue
        role = classify_material_role(mat.name, base_img.name)
        alpha_policy = get_alpha_policy(role, mat.name, base_img.name)
        job_key = f"{base_img.name}::{alpha_policy}"
        if job_key not in image_jobs:
            image_jobs[job_key] = (
                base_img,
                role,
                alpha_policy,
                f"{mat.name} {base_img.name}",
            )

    manifest = {}

    for job_key, (img_obj, role, alpha_policy, combined_name) in image_jobs.items():
        disk_path = resolve_disk_path(img_obj, tex_dirs, tex_index=tex_index)
        try:
            if (
                not disk_path or not os.path.exists(disk_path)
            ) and not img_obj.has_data:
                try:
                    img_obj.reload()
                except Exception:
                    pass
            pil_img = image_to_pil_rgba(img_obj, disk_path)
            if pil_img is None:
                print(
                    f"   ⚠️ [贴图缺失] 无法加载 {job_key} (filepath={img_obj.filepath!r}, mat_hint={combined_name})"
                )
                continue

            pil_img = resize_if_needed(pil_img, max_tex_size)

            arr = np.array(pil_img, dtype=np.float32)
            raw_trans_pct = float(np.mean(arr[:, :, 3] < 20.0)) * 100.0
            raw_black_pct = float(np.mean(np.max(arr[:, :, :3], axis=2) < 18.0)) * 100.0

            is_plus = alpha_policy == "plus"
            arr = maybe_synthesize_black_bg_alpha(arr, role, is_plus=is_plus)
            has_alpha, alpha_mode = detect_alpha_mode(
                arr, role, combined_name, is_plus=is_plus
            )

            short_hash = hashlib.md5(job_key.encode("utf-8")).hexdigest()[:6]
            raw_base = os.path.basename(disk_path) if disk_path else img_obj.name
            safe_stem = re.sub(r"[^a-zA-Z0-9_\-]", "_", os.path.splitext(raw_base)[0])
            key_prefix = f"zzz_{safe_stem}_{alpha_policy}_{short_hash}"

            if has_alpha:
                lit_fname = f"{key_prefix}_lit.png"
                lit_path = os.path.join(temp_dir, lit_fname)
                Image.fromarray(
                    np.clip(arr, 0, 255).astype(np.uint8), mode="RGBA"
                ).save(lit_path, format="PNG", optimize=True, compress_level=9)
            else:
                lit_fname = f"{key_prefix}_lit.jpg"
                lit_path = os.path.join(temp_dir, lit_fname)
                Image.fromarray(
                    np.clip(arr[:, :, :3], 0, 255).astype(np.uint8), mode="RGB"
                ).save(
                    lit_path, format="JPEG", quality=90, subsampling=0, optimize=True
                )

            emi_fname = None
            emi_disk = find_emissive_counterpart(
                disk_path, tex_dirs, tex_index=tex_index
            )
            if emi_disk and os.path.exists(emi_disk):
                emi_img = resize_if_needed(
                    Image.open(emi_disk).convert("RGB"), max_tex_size
                )
                emi_fname = f"{key_prefix}_emi.jpg"
                emi_img.save(
                    os.path.join(temp_dir, emi_fname),
                    format="JPEG",
                    quality=88,
                    subsampling=0,
                    optimize=True,
                )

            entry = {
                "role": role,
                "alpha_policy": alpha_policy,
                "lit": lit_fname,
                "emissive": emi_fname,
                "has_alpha": has_alpha,
                "alpha_mode": alpha_mode,
            }
            manifest[job_key] = entry
            manifest.setdefault(img_obj.name, entry)

            size_kb = os.path.getsize(lit_path) // 1024
            print(
                f"   ✅ [贴图压缩] {raw_base} ({alpha_policy}, 透明={raw_trans_pct:.0f}%, 纯黑={raw_black_pct:.0f}%) "
                f"-> {lit_fname} ({size_kb} KB, role={role}, alpha={alpha_mode})"
            )
        except Exception as e:
            print(f"   ⚠️ 贴图 [{job_key}] 处理失败: {e}")

    manifest_path = os.path.join(temp_dir, "zzz_bake_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"✅ [步骤 4] 贴图优化完成！共处理 {len(image_jobs)} 份烘焙贴图。")
