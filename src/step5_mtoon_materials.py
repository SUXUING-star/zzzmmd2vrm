import json
import os

import bpy
import numpy as np

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
                old_img = bpy.data.images.get(os.path.basename(fpath))
                if old_img is not None:
                    try:
                        bpy.data.images.remove(old_img, do_unlink=True)
                    except Exception:
                        pass
                img = bpy.data.images.load(fpath, check_existing=False)
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
    double_sided = False

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
    dr, dg, db = diffuse_rgb
    ar, ag, ab = ambient_rgb
    if max(dr, dg, db) <= 0.02:
        if max(ar, ag, ab) <= 0.05 or role in ("lash_brow",):
            return (dr, dg, db)
        return (min(1.0, ar), min(1.0, ag), min(1.0, ab))

    return (
        min(1.0, dr + ar * 0.5),
        min(1.0, dg + ag * 0.5),
        min(1.0, db + ab * 0.5),
    )


def get_shade_tint_tuple(shading_role):
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


def sample_uv_diagnostics(target_names):
    """采样重点材质在 UVMap 上的实际不透明像素 RGB 与 Alpha 均值"""
    for obj in bpy.data.objects:
        if obj.type != "MESH" or not obj.data or not obj.data.polygons:
            continue
        mesh = obj.data
        if not mesh.uv_layers:
            continue
        uv_data = mesh.uv_layers.active.data
        slot_by_name = {
            m.name: idx
            for idx, m in enumerate(mesh.materials)
            if m and m.name in target_names
        }
        if not slot_by_name:
            continue

        img_cache = {}
        for s_name, s_idx in slot_by_name.items():
            mat = mesh.materials[s_idx]
            gltf = mat.vrm_addon_extension.mtoon1
            img = gltf.pbr_metallic_roughness.base_color_texture.index.source
            us, vs = [], []
            for poly in mesh.polygons:
                if poly.material_index != s_idx:
                    continue
                for l_idx in poly.loop_indices:
                    u, v = uv_data[l_idx].uv
                    us.append(u)
                    vs.append(v)
            if not us:
                continue
            uv_box = f"u=[{min(us):.2f},{max(us):.2f}], v=[{min(vs):.2f},{max(vs):.2f}]"
            if img and img.has_data and img.size[0] > 0 and img.size[1] > 0:
                if img.name not in img_cache:
                    w, h = img.size[0], img.size[1]
                    arr = np.array(img.pixels[:], dtype=np.float32).reshape((h, w, 4))
                    img_cache[img.name] = (w, h, arr)
                w, h, arr = img_cache[img.name]
                step = max(1, len(us) // 256)
                su = np.mod(np.array(us[::step]), 1.0)
                sv = np.mod(np.array(vs[::step]), 1.0)
                ix = np.clip((su * (w - 1)).astype(np.int32), 0, w - 1)
                iy = np.clip((sv * (h - 1)).astype(np.int32), 0, h - 1)
                px = arr[iy, ix]
                m_a = float(px[:, 3].mean())
                vis = px[:, 3] >= 0.1
                vis_rgb = (
                    px[vis, :3].mean(axis=0) if np.any(vis) else px[:, :3].mean(axis=0)
                )
                print(
                    f"   🔬 [UV采样] {s_name} | img={img.name}({w}x{h}) | alpha_mode={gltf.alpha_mode} | "
                    f"{uv_box} | 可见RGB=({vis_rgb[0]:.2f},{vis_rgb[1]:.2f},{vis_rgb[2]:.2f}), A均值={m_a:.2f}"
                )
            else:
                print(
                    f"   🔬 [UV采样] {s_name} | 无贴图 | alpha_mode={gltf.alpha_mode} | {uv_box}"
                )


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
    uv_diag_targets = set()
    loaded_cache = {}

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
        alpha_policy = get_alpha_policy(role, mat.name, orig_img_name)

        if role in ("face", "skin"):
            shading_role = "face"
        elif role in ("eye", "fx_overlay", "hair", "metal", "mouth_inner"):
            shading_role = role
        else:
            shading_role = role

        manifest_key = f"{orig_img_name}::{alpha_policy}" if orig_img_name else ""
        info = manifest.get(manifest_key) or manifest.get(orig_img_name, {})
        is_dead_spa = bool(info.get("is_dead_spa_overlay", False))

        # 动态识别仅用于 MMD .spa 提亮的叠加层（如 髮+ 无 Base 贴图或复用 0% 透明底图，或漫反射为纯黑）
        is_no_base_plus_spa = alpha_policy == "plus" and orig_img is None
        is_untextured_plus_spa = (
            alpha_policy == "plus"
            and orig_img is not None
            and not info.get("has_alpha", False)
            and mat_alpha >= 0.99
        )
        is_black_diffuse_plus = (
            alpha_policy == "plus"
            and max(diffuse_rgb) <= 0.05
            and max(ambient_rgb) <= 0.05
        )
        hide_overlay = (
            is_dead_spa
            or is_no_base_plus_spa
            or is_untextured_plus_spa
            or is_black_diffuse_plus
        )
        eff_alpha = 0.0 if hide_overlay else mat_alpha

        lit_fname = info.get("lit")
        if lit_fname:
            if lit_fname not in loaded_cache:
                loaded_cache[lit_fname] = load_baked_img(tex_dirs, lit_fname, "sRGB")
            lit_img = loaded_cache[lit_fname]
        else:
            lit_img = None

        if lit_img is None and orig_img is not None:
            if not orig_img.has_data or orig_img.size[0] <= 0 or orig_img.size[1] <= 0:
                try:
                    orig_img.reload()
                except Exception:
                    pass
            if orig_img.has_data and orig_img.size[0] > 0 and orig_img.size[1] > 0:
                lit_img = orig_img
            else:
                print(
                    f"   ⚠️ [空贴图拦截] 材质 [{mat.name}] 的原图 [{orig_img.name}] 无像素数据 (filepath={orig_img.filepath!r})，已回退纯色防止变黑！"
                )

        emi_fname = info.get("emissive")
        if emi_fname:
            if emi_fname not in loaded_cache:
                loaded_cache[emi_fname] = load_baked_img(tex_dirs, emi_fname, "sRGB")
            emissive_img = loaded_cache[emi_fname]
        else:
            emissive_img = None

        ext = getattr(mat, "vrm_addon_extension", None)
        if ext is None or not hasattr(ext, "mtoon1"):
            print(f"   ⚠️ 材质 [{mat.name}] 未找到 vrm_addon_extension.mtoon1")
            continue

        mat.use_nodes = True
        gltf = ext.mtoon1
        gltf.enabled = True
        gltf.double_sided = (
            True if role in ("lash_brow", "fx_overlay") else is_double_sided
        )

        if lit_img:
            try:
                lit_img.pack()
            except Exception:
                pass
            gltf.pbr_metallic_roughness.base_color_factor = (1.0, 1.0, 1.0, eff_alpha)
            gltf.pbr_metallic_roughness.base_color_texture.index.source = lit_img
        else:
            ur, ug, ub = resolve_untextured_color(diffuse_rgb, ambient_rgb, role)
            gltf.pbr_metallic_roughness.base_color_factor = (ur, ug, ub, eff_alpha)

        if emissive_img and not hide_overlay:
            gltf.emissive_texture.index.source = emissive_img
            gltf.emissive_factor = (1.0, 1.0, 1.0)
        else:
            gltf.emissive_factor = (0.0, 0.0, 0.0)

        suggested_alpha = info.get("alpha_mode", "OPAQUE")
        if (
            hide_overlay
            or eff_alpha < 0.99
            or suggested_alpha == "BLEND"
            or (role == "fx_overlay" and info.get("has_alpha"))
        ):
            gltf.alpha_mode = "BLEND"
        elif suggested_alpha == "MASK" or (
            role in ("lash_brow", "cutout_cloth") and info.get("has_alpha")
        ):
            gltf.alpha_mode = "MASK"
            gltf.alpha_cutoff = 0.5
        else:
            gltf.alpha_mode = "OPAQUE"

        if hasattr(mat, "surface_render_method") and gltf.alpha_mode != "BLEND":
            if mat.surface_render_method == "BLENDED":
                mat.surface_render_method = "DITHERED"

        mtoon = gltf.extensions.vrmc_materials_mtoon
        # 紧贴表面的贴片(fx_overlay / plus)关闭深度写入；实心半透 3D 部件(如水晶鞋跟)开启深度写入防止透视消失
        if gltf.alpha_mode == "BLEND":
            mtoon.transparent_with_z_write = (
                not hide_overlay
                and role != "fx_overlay"
                and alpha_policy != "plus"
                and eff_alpha > 0.01
            )
        else:
            mtoon.transparent_with_z_write = False
        mtoon.render_queue_offset_number = 1 if gltf.alpha_mode == "BLEND" else 0

        tint = get_shade_tint_tuple(shading_role)
        if lit_img:
            mtoon.shade_multiply_texture.index.source = lit_img
            mtoon.shade_color_factor = tint
        else:
            ur, ug, ub = resolve_untextured_color(diffuse_rgb, ambient_rgb, role)
            mtoon.shade_color_factor = (ur * tint[0], ug * tint[1], ub * tint[2])

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

        if (
            gltf.alpha_mode != "OPAQUE"
            or hide_overlay
            or "+" in mat.name
            or any(k in mat.name for k in ("体", "鞋", "脚", "髮", "发"))
        ):
            print(
                f"   🎨 [材质诊断] {mat.name} | img={orig_img_name or 'None'} | role={role}, policy={alpha_policy} | "
                f"alpha_mode={gltf.alpha_mode}, eff_alpha={eff_alpha:.2f}, "
                f"z_write={mtoon.transparent_with_z_write}, hide={hide_overlay}"
            )

        if any(k in mat.name for k in ("手", "肌", "衣金属+", "鞋+")):
            uv_diag_targets.add(mat.name)

        converted += 1

    if uv_diag_targets:
        sample_uv_diagnostics(uv_diag_targets)

    for img in list(bpy.data.images):
        if not img.has_data or img.size[0] <= 0 or img.size[1] <= 0:
            try:
                bpy.data.images.remove(img, do_unlink=True)
            except Exception:
                pass

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
