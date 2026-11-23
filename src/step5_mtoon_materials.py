import json
import os

import bpy

from step0_mat_map import classify_material_role, extract_material_base_image
from step4_bake_texture import get_alpha_policy, normalize_texture_dirs


def load_baked_img(texture_dirs, filename, colorspace="sRGB"):
    if not texture_dirs or not filename:
        return None
    if isinstance(texture_dirs, str):
        texture_dirs = [texture_dirs]

    for td in texture_dirs:
        for fpath in (
            os.path.join(td, "vrm_temp_baked", filename),
            os.path.join(td, filename),
        ):
            if os.path.exists(fpath):
                img = bpy.data.images.load(fpath, check_existing=True)
                try:
                    img.colorspace_settings.name = colorspace
                except Exception:
                    pass
                try:
                    img.pack()
                except Exception:
                    pass
                return img
    return None


def extract_mmd_or_bsdf_props(mat):
    diffuse = (1.0, 1.0, 1.0)
    ambient = (0.5, 0.5, 0.5)
    alpha = 1.0
    double_sided = True

    mmd_mat = getattr(mat, "mmd_material", None)
    if mmd_mat:
        try:
            dc = mmd_mat.diffuse_color
            diffuse = (float(dc[0]), float(dc[1]), float(dc[2]))
            ac = getattr(mmd_mat, "ambient_color", (0.5, 0.5, 0.5))
            ambient = (float(ac[0]), float(ac[1]), float(ac[2]))
            alpha = float(mmd_mat.alpha)
            double_sided = bool(mmd_mat.is_double_sided)
            return diffuse, ambient, alpha, double_sided
        except Exception:
            pass

    if mat.use_nodes and mat.node_tree:
        for node in mat.node_tree.nodes:
            if node.type == "BSDF_PRINCIPLED":
                bc = node.inputs.get("Base Color")
                if bc and not bc.is_linked:
                    c = bc.default_value
                    diffuse = (float(c[0]), float(c[1]), float(c[2]))
                al = node.inputs.get("Alpha")
                if al and not al.is_linked:
                    alpha = float(al.default_value)
                break

    return diffuse, ambient, alpha, double_sided


def resolve_untextured_color(diffuse_rgb, ambient_rgb, role):
    """无贴图材质（如目白/齿/目光/睫/口线）：补偿 MMD 暗 diffuse+ambient，同时保住纯黑线条"""
    dr, dg, db = diffuse_rgb
    ar, ag, ab = ambient_rgb
    if max(dr, dg, db) <= 0.02:
        # 睫毛/口线/二重等若 diffuse 与 ambient 都是黑的，保持纯黑不要变成白色
        if max(ar, ag, ab) <= 0.05 or role in ("lash_brow",):
            return (dr, dg, db)
        return (min(1.0, ar), min(1.0, ag), min(1.0, ab))

    # MMD 无贴图亮部近似为 diffuse + ambient 补偿，防止目白/齿发灰
    return (
        min(1.0, dr + ar * 0.5),
        min(1.0, dg + ag * 0.5),
        min(1.0, db + ab * 0.5),
    )


def get_shade_tint_tuple(shading_role):
    """参考 umammd 暖肉色皮肤阴影与通透微紫灰暗部，避免发灰发脏"""
    if shading_role in ("face", "skin"):
        return (0.92, 0.76, 0.74)
    if shading_role in ("eye", "fx_overlay"):
        return (0.95, 0.94, 0.97)
    if shading_role == "hair":
        return (0.80, 0.78, 0.85)
    if shading_role == "metal":
        return (0.76, 0.78, 0.85)
    if shading_role == "mouth_inner":
        return (0.86, 0.75, 0.75)
    return (0.80, 0.78, 0.85)


def run(texture_dir):
    print(
        ">>> [步骤 5] 绑定 VRM MToon 1.0 材质 (融合 umammd 亮部调色与 ZZZ 多目录贴图)..."
    )
    tex_dirs = normalize_texture_dirs(texture_dir)

    manifest = {}
    for td in tex_dirs:
        for fname in ("zzz_bake_manifest.json", "ww_bake_manifest.json"):
            m_path = os.path.join(td, "vrm_temp_baked", fname)
            if os.path.exists(m_path):
                with open(m_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                break
        if manifest:
            break

    converted = 0
    total = 0

    for mat in bpy.data.materials:
        if not mat or getattr(mat, "users", 0) == 0:
            continue
        if getattr(mat, "is_grease_pencil", False):
            continue
        total += 1

        orig_img = extract_material_base_image(mat)
        diffuse_rgb, ambient_rgb, mat_alpha, is_double_sided = (
            extract_mmd_or_bsdf_props(mat)
        )
        orig_img_name = orig_img.name if orig_img else ""
        role = classify_material_role(mat.name, orig_img_name)
        alpha_policy = get_alpha_policy(role)

        if role in ("face", "skin"):
            shading_role = "face"
        elif role in ("eye", "fx_overlay", "hair", "metal", "mouth_inner"):
            shading_role = role
        else:
            shading_role = role

        manifest_key = f"{orig_img_name}::{alpha_policy}" if orig_img_name else ""
        info = manifest.get(manifest_key) or manifest.get(orig_img_name, {})
        lit_img = load_baked_img(tex_dirs, info.get("lit"), "sRGB") or orig_img
        emissive_img = load_baked_img(tex_dirs, info.get("emissive"), "sRGB")

        ext = getattr(mat, "vrm_addon_extension", None)
        if ext is None or not hasattr(ext, "mtoon1"):
            print(f"   ⚠️ 材质 [{mat.name}] 未找到 vrm_addon_extension.mtoon1")
            continue

        mat.use_nodes = True
        gltf = ext.mtoon1
        gltf.enabled = True
        gltf.double_sided = (
            True
            if role in ("lash_brow", "cutout_cloth", "fx_overlay", "hair")
            else is_double_sided
        )

        # 核心修复（同 umammd）：只要存在贴图，base_color_factor 必须保持纯白 (1, 1, 1, alpha)，
        # 避免 MMD 默认偏暗/偏灰的 diffuse_color 直接把贴图乘黑！
        if lit_img:
            try:
                lit_img.pack()
            except Exception:
                pass
            gltf.pbr_metallic_roughness.base_color_factor = (1.0, 1.0, 1.0, mat_alpha)
            gltf.pbr_metallic_roughness.base_color_texture.index.source = lit_img
        else:
            ur, ug, ub = resolve_untextured_color(diffuse_rgb, ambient_rgb, role)
            gltf.pbr_metallic_roughness.base_color_factor = (ur, ug, ub, mat_alpha)

        if emissive_img:
            gltf.emissive_texture.index.source = emissive_img
            gltf.emissive_factor = (1.0, 1.0, 1.0)
        else:
            gltf.emissive_factor = (0.0, 0.0, 0.0)

        suggested_alpha = info.get("alpha_mode", "OPAQUE")
        if mat_alpha < 0.99 or role == "fx_overlay" or suggested_alpha == "BLEND":
            gltf.alpha_mode = "BLEND"
        elif role in ("lash_brow", "cutout_cloth") and (
            suggested_alpha == "MASK" or info.get("has_alpha")
        ):
            gltf.alpha_mode = "MASK"
            gltf.alpha_cutoff = 0.5
        else:
            gltf.alpha_mode = "OPAQUE"

        mtoon = gltf.extensions.vrmc_materials_mtoon
        mtoon.transparent_with_z_write = False
        mtoon.render_queue_offset_number = 1 if gltf.alpha_mode == "BLEND" else 0

        # 暗部颜色同样不再乘偏暗的 diffuse_rgb，直接套用通透阴影 tint
        tint = get_shade_tint_tuple(shading_role)
        if lit_img:
            mtoon.shade_multiply_texture.index.source = lit_img
            mtoon.shade_color_factor = tint
        else:
            ur, ug, ub = resolve_untextured_color(diffuse_rgb, ambient_rgb, role)
            mtoon.shade_color_factor = (ur * tint[0], ug * tint[1], ub * tint[2])

        # 沿用 umammd 明暗交界与 GI 均等化调色参数（整体提亮通透）
        if shading_role in ("face", "skin"):
            mtoon.shading_shift_factor = 0.12
            mtoon.shading_toony_factor = 0.22
            mtoon.gi_equalization_factor = 0.90
        elif shading_role in ("eye", "fx_overlay"):
            mtoon.shading_shift_factor = 0.0
            mtoon.shading_toony_factor = 0.15
            mtoon.gi_equalization_factor = 0.95
        elif shading_role == "hair":
            mtoon.shading_shift_factor = 0.0
            mtoon.shading_toony_factor = 0.55
            mtoon.gi_equalization_factor = 0.75
        elif shading_role == "metal":
            mtoon.shading_shift_factor = 0.0
            mtoon.shading_toony_factor = 0.60
            mtoon.gi_equalization_factor = 0.70
        else:
            mtoon.shading_shift_factor = 0.05
            mtoon.shading_toony_factor = 0.38
            mtoon.gi_equalization_factor = 0.80

        mtoon.outline_width_mode = "none"
        if mat.node_tree:
            mat.node_tree.update_tag()
        converted += 1

    try:
        bpy.ops.outliner.orphans_purge(
            do_local_ids=True, do_linked_ids=True, do_recursive=True
        )
    except Exception:
        pass
    for img in list(bpy.data.images):
        if img.users == 0:
            bpy.data.images.remove(img)

    print(f"✅ [步骤 5] MToon 1.0 材质转换完成：{converted}/{total} 个材质。")
