# main.py
import os

import bpy

import step0_clean
import step1_skeleton
import step2_shapekey
import step3_bind_vrm
import step4_bake_texture
import step5_mtoon_materials
import step6_metadata
import step_vrma

_TEX_DIR_EXACT = {
    "tex",
    "texture",
    "textures",
    "texture2d",
    "img",
    "images",
    "map",
    "maps",
}
_SKIP_DIRS = {"spa", "sph", "toon", "toons", "vrm_temp_baked", "__pycache__"}


def _split_dir_config(val):
    if not val:
        return []
    if isinstance(val, (list, tuple, set)):
        items = [str(x).strip() for x in val if x and str(x).strip()]
    else:
        # 兼容用 ; 传多个路径
        items = [x.strip() for x in str(val).split(";") if x.strip()]
    return items


def discover_texture_dirs(config, blend_dir):
    """自动搜集所有贴图目录（支持多目录并行存在 + 自动探测 tex/Texture2D/根目录/已引用路径）"""
    candidates = []

    for raw in _split_dir_config(config.get("texture_dirs")) + _split_dir_config(
        config.get("texture_dir")
    ):
        p = raw if os.path.isabs(raw) else os.path.abspath(os.path.join(blend_dir, raw))
        candidates.append(p)

    # 自动扫描 blend_dir 及其下所有 tex* / Texture2D / textures 目录（不 break，全收）
    scan_roots = [blend_dir]
    for p in list(candidates):
        parent = os.path.dirname(p)
        if parent and os.path.isdir(parent):
            scan_roots.append(parent)

    for root in scan_roots:
        if not root or not os.path.isdir(root):
            continue
        try:
            for entry in sorted(os.listdir(root)):
                low = entry.lower()
                if low in _SKIP_DIRS:
                    continue
                full = os.path.join(root, entry)
                if not os.path.isdir(full):
                    continue
                if low in _TEX_DIR_EXACT or low.startswith(("tex", "texture")):
                    candidates.append(full)
        except OSError:
            pass

    # 把模型根目录也兜底加入（放最后，避免根目录下 toon bmp 抢优先级）
    if blend_dir and os.path.isdir(blend_dir):
        candidates.append(blend_dir)

    # 再补充 Blender 当前工程里贴图实际指向的目录
    for img in bpy.data.images:
        fp = (img.filepath or "").replace("\\", "/")
        if not fp:
            continue
        abs_fp = bpy.path.abspath(fp)
        d = os.path.dirname(abs_fp)
        if d and os.path.isdir(d) and os.path.basename(d).lower() not in _SKIP_DIRS:
            candidates.append(d)

    unique_dirs = []
    seen = set()
    for d in candidates:
        norm = os.path.normcase(os.path.abspath(d))
        if norm not in seen and os.path.isdir(d):
            seen.add(norm)
            unique_dirs.append(os.path.abspath(d))

    return unique_dirs


def resolve_config_paths(config):
    blend_dir = bpy.path.abspath("//")

    models_dir = config.get("models_dir") or "."
    if not os.path.isabs(models_dir):
        config["models_dir"] = os.path.abspath(os.path.join(blend_dir, models_dir))

    blenders_dir = config.get("blenders_dir") or "."
    if not os.path.isabs(blenders_dir):
        config["blenders_dir"] = os.path.abspath(os.path.join(blend_dir, blenders_dir))

    tex_dirs = discover_texture_dirs(config, blend_dir)
    config["texture_dirs"] = tex_dirs
    # 兼容仍读取 texture_dir 的外部逻辑：指向主贴图目录（用于存放 vrm_temp_baked）
    config["texture_dir"] = tex_dirs[0] if tex_dirs else blend_dir

    meta = config.get("metadata", {})
    thumb = meta.get("thumbnail_path", "")
    if thumb and not os.path.isabs(thumb):
        meta["thumbnail_path"] = os.path.abspath(os.path.join(blend_dir, thumb))

    vrma = config.get("vrma_path", "")
    if vrma and not os.path.isabs(vrma):
        config["vrma_path"] = os.path.abspath(os.path.join(blend_dir, vrma))

    if isinstance(config.get("vrma_paths"), list):
        config["vrma_paths"] = [
            os.path.abspath(os.path.join(blend_dir, p))
            if p and not os.path.isabs(p)
            else p
            for p in config["vrma_paths"]
        ]

    return config


def run(config):
    config = resolve_config_paths(config)

    jp = config["jp_name"]
    en = config["en_name"]
    tex_dirs = config["texture_dirs"]
    max_tex_size = int(config.get("max_tex_size", 1024))
    max_bones = int(config.get("max_bones", 240))
    ground_to_zero = bool(config.get("ground_to_zero", True))
    fix_high_heel_rig = bool(config.get("fix_high_heel_rig", True))

    print(f"\n>>> [zzzmmd2vrm] 开始处理模型: {jp} -> {en}")
    print(f"   [路径] VRM 导出目录: {config['models_dir']}")
    print(
        f"   [路径] 贴图搜索路径({len(tex_dirs)}个): {tex_dirs} "
        f"(max_tex_size={max_tex_size}, max_bones={max_bones}, ground_to_zero={ground_to_zero}, fix_high_heel_rig={fix_high_heel_rig})"
    )

    step0_clean.run()
    step1_skeleton.run(
        jp,
        en,
        max_bones=max_bones,
        ground_to_zero=ground_to_zero,
        fix_high_heel_rig=fix_high_heel_rig,
    )
    step2_shapekey.run()
    step3_bind_vrm.run()
    step4_bake_texture.run(tex_dirs, max_tex_size=max_tex_size)
    step5_mtoon_materials.run(tex_dirs)
    step_vrma.run(config)
    step6_metadata.run(config)
