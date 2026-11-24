import math
from collections import defaultdict

import bmesh
import bpy
from mathutils import Vector, kdtree

from step0_mat_map import classify_surface_for_normals, extract_material_base_image
from step0_springbone import snapshot_mmd_physics_to_armature


def snapshot_custom_normals(mesh):
    if hasattr(mesh, "corner_normals"):
        loop_normals = [Vector(cn.vector) for cn in mesh.corner_normals]
    else:
        mesh.calc_normals_split()
        loop_normals = [Vector(l.normal) for l in mesh.loops]

    verts = mesh.vertices
    loops = mesh.loops
    kd = kdtree.KDTree(len(loops))

    for poly in mesh.polygons:
        p_no = poly.normal
        for l_idx in poly.loop_indices:
            v_co = verts[loops[l_idx].vertex_index].co
            kd.insert(v_co + p_no * 0.001, l_idx)

    kd.balance()
    return kd, loop_normals


def restore_custom_normals(mesh, kd, orig_loop_normals):
    if hasattr(mesh, "use_auto_smooth"):
        mesh.use_auto_smooth = True

    verts = mesh.vertices
    loops = mesh.loops
    restored = [(0.0, 0.0, 1.0)] * len(loops)

    for poly in mesh.polygons:
        p_no = poly.normal
        for l_idx in poly.loop_indices:
            v_co = verts[loops[l_idx].vertex_index].co
            _, orig_idx, _ = kd.find(v_co + p_no * 0.001)
            restored[l_idx] = tuple(orig_loop_normals[orig_idx])

    mesh.normals_split_custom_set(restored)
    mesh.update()


def get_head_bone_local_z(obj):
    armature = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
    if not armature:
        return None
    for bone in armature.data.bones:
        b_low = bone.name.lower()
        if "head" in b_low or "頭" in bone.name or "头" in bone.name:
            world_pos = armature.matrix_world @ bone.head_local
            local_pos = obj.matrix_world.inverted() @ world_pos
            return local_pos.z
    return None


def compute_welded_smooth_normals(
    orig_obj, poly_indices, smooth_passes=3, crease_deg=70.0
):
    """在内存 bmesh 中按同材质焊接改模未合并裂缝，做纯法线空间平滑（不移动顶点，避免嘴角/背部挤压脏影）"""
    orig_mesh = orig_obj.data
    target_slots = {orig_mesh.polygons[p].material_index for p in poly_indices}
    cos_limit = math.cos(math.radians(crease_deg))

    group_verts = set()
    vert_slot = {}
    orig_vert_acc = defaultdict(Vector)
    for p_idx in poly_indices:
        poly = orig_mesh.polygons[p_idx]
        p_no = poly.normal
        valid_no = all(math.isfinite(x) for x in p_no) and p_no.length_squared > 1e-12
        area_n = p_no * max(poly.area, 1e-8) if valid_no else Vector((0.0, 0.0, 0.0))
        for v_idx in poly.vertices:
            group_verts.add(v_idx)
            vert_slot.setdefault(v_idx, poly.material_index)
            orig_vert_acc[v_idx] += area_n

    bm = bmesh.new()
    bm.from_mesh(orig_mesh)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    welded_cnt = 0
    try:
        bm_vert_slots = defaultdict(set)
        bm_vert_acc = defaultdict(Vector)
        for f in bm.faces:
            is_tgt = f.material_index in target_slots
            f_no = f.normal
            valid_no = (
                all(math.isfinite(x) for x in f_no) and f_no.length_squared > 1e-12
            )
            area_n = (
                f_no * max(f.calc_area(), 1e-8) if valid_no else Vector((0.0, 0.0, 0.0))
            )
            for v in f.verts:
                bm_vert_slots[v.index].add(f.material_index if is_tgt else -1)
                if is_tgt:
                    bm_vert_acc[v.index] += area_n

        # 仅在同一目标材质槽内部焊接间距 <=0.15mm 且朝向一致的裂缝顶点
        by_slot = defaultdict(list)
        for v in bm.verts:
            s_set = bm_vert_slots.get(v.index)
            if s_set and len(s_set) == 1:
                s = next(iter(s_set))
                if s in target_slots:
                    by_slot[s].append(v)

        targetmap = {}
        for s, v_list in by_slot.items():
            if len(v_list) < 2:
                continue
            kd_weld = kdtree.KDTree(len(v_list))
            for idx, v in enumerate(v_list):
                kd_weld.insert(v.co, idx)
            kd_weld.balance()

            visited = set()
            for idx, v in enumerate(v_list):
                if idx in visited:
                    continue
                n_base = bm_vert_acc[v.index]
                if n_base.length_squared <= 1e-12:
                    continue
                n_base_u = n_base.normalized()
                hits = kd_weld.find_range(v.co, 0.00015)
                if len(hits) <= 1:
                    continue
                visited.add(idx)
                for _, other_idx, _ in hits:
                    if other_idx == idx or other_idx in visited:
                        continue
                    ov = v_list[other_idx]
                    n_other = bm_vert_acc[ov.index]
                    if (
                        n_other.length_squared > 1e-12
                        and n_base_u.dot(n_other.normalized()) > 0.30
                    ):
                        targetmap[ov] = v
                        visited.add(other_idx)

        if targetmap:
            welded_cnt = len(targetmap)
            bmesh.ops.weld_verts(bm, targetmap=targetmap)

        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        bm.edges.ensure_lookup_table()

        # 在焊接后的拓扑上重新计算面积加权顶点法线，并建立单环邻接图
        w_acc = defaultdict(Vector)
        w_slots = defaultdict(set)
        w_neighbors = defaultdict(set)
        for f in bm.faces:
            if f.material_index not in target_slots:
                continue
            f_no = f.normal
            if not all(math.isfinite(x) for x in f_no) or f_no.length_squared < 1e-12:
                continue
            area_n = f_no * max(f.calc_area(), 1e-8)
            f_verts = [v.index for v in f.verts]
            for v_idx in f_verts:
                w_acc[v_idx] += area_n
                w_slots[v_idx].add(f.material_index)
            for e in f.edges:
                va, vb = e.verts[0].index, e.verts[1].index
                w_neighbors[va].add(vb)
                w_neighbors[vb].add(va)

        w_norm = {}
        for v_idx, acc in w_acc.items():
            if acc.length_squared > 1e-12:
                w_norm[v_idx] = acc.normalized()

        # 纯法线空间拉普拉斯平滑（受角度保护，不移顶点，彻底抚平改模凹凸法线）
        for _ in range(smooth_passes):
            nxt = {}
            for v_idx, n_cur in w_norm.items():
                nbs = w_neighbors.get(v_idx)
                if not nbs:
                    nxt[v_idx] = n_cur
                    continue
                acc_nb = Vector((0.0, 0.0, 0.0))
                cnt_nb = 0
                for nb in nbs:
                    n_nb = w_norm.get(nb)
                    if n_nb is not None and n_cur.dot(n_nb) >= cos_limit:
                        acc_nb += n_nb
                        cnt_nb += 1
                if cnt_nb > 0 and acc_nb.length_squared > 1e-12:
                    blended = n_cur * 0.45 + acc_nb.normalized() * 0.55
                    nxt[v_idx] = (
                        blended.normalized()
                        if blended.length_squared > 1e-12
                        else n_cur
                    )
                else:
                    nxt[v_idx] = n_cur
            w_norm = nxt

        # 计算原始坐标兜底法线
        slot_orig_list = defaultdict(list)
        for v_idx in group_verts:
            slot_orig_list[vert_slot[v_idx]].append(v_idx)

        clean_fallback = {}
        for s, v_list in slot_orig_list.items():
            kd_orig = kdtree.KDTree(len(v_list))
            for idx, v_idx in enumerate(v_list):
                kd_orig.insert(orig_mesh.vertices[v_idx].co, idx)
            kd_orig.balance()
            for v_idx in v_list:
                n_base = orig_vert_acc[v_idx]
                if n_base.length_squared <= 1e-12:
                    on = Vector(orig_mesh.vertices[v_idx].normal)
                    clean_fallback[v_idx] = (
                        on.normalized()
                        if on.length_squared > 1e-12
                        else Vector((0.0, 0.0, 1.0))
                    )
                    continue
                n_u = n_base.normalized()
                merged = n_base.copy()
                for _, o_idx, _ in kd_orig.find_range(
                    orig_mesh.vertices[v_idx].co, 0.00015
                ):
                    ov = v_list[o_idx]
                    if ov == v_idx:
                        continue
                    n_o = orig_vert_acc[ov]
                    if n_o.length_squared > 1e-12 and n_u.dot(n_o.normalized()) > 0.30:
                        merged += n_o
                clean_fallback[v_idx] = (
                    merged.normalized() if merged.length_squared > 1e-12 else n_u
                )

        # 用焊接后 bmesh 坐标建 KDTree 映射回 orig_mesh 顶点
        slot_kd = {}
        slot_vecs = {}
        for s in target_slots:
            p_verts = [
                v_idx
                for v_idx, s_set in w_slots.items()
                if s in s_set and v_idx in w_norm
            ]
            if not p_verts:
                continue
            kd_p = kdtree.KDTree(len(p_verts))
            for i_p, v_idx in enumerate(p_verts):
                kd_p.insert(bm.verts[v_idx].co, i_p)
            kd_p.balance()
            slot_kd[s] = kd_p
            slot_vecs[s] = [(v_idx, w_norm[v_idx]) for v_idx in p_verts]

        result = {}
        for v_idx in group_verts:
            fb_n = clean_fallback[v_idx]
            s = vert_slot[v_idx]
            kd_p = slot_kd.get(s)
            if kd_p is None:
                result[v_idx] = fb_n
                continue
            co = orig_mesh.vertices[v_idx].co
            best_n = None
            best_dot = -1.0
            for _, i_p, _ in kd_p.find_range(co, 0.0002):
                _, pn = slot_vecs[s][i_p]
                d = fb_n.dot(pn)
                if d > 0.15 and d > best_dot:
                    best_dot = d
                    best_n = pn
            if best_n is None:
                _, i_p, _ = kd_p.find(co)
                _, pn = slot_vecs[s][i_p]
                best_n = pn if fb_n.dot(pn) > 0.15 else fb_n
            result[v_idx] = best_n

        return result, clean_fallback, welded_cnt
    finally:
        bm.free()


def apply_face_ellipsoid_blend(orig_obj, face_vert_normals, blend=0.50):
    if not face_vert_normals:
        return face_vert_normals

    mesh = orig_obj.data
    head_z = get_head_bone_local_z(orig_obj)

    target_verts = [
        v_idx
        for v_idx in face_vert_normals
        if head_z is None or mesh.vertices[v_idx].co.z >= head_z - 0.02
    ]
    if not target_verts:
        return face_vert_normals

    coords = [mesh.vertices[v].co for v in target_verts]
    xs = sorted(c.x for c in coords)
    ys = sorted(c.y for c in coords)
    zs = sorted(c.z for c in coords)
    n_c = len(coords)
    i_lo = min(n_c - 1, max(0, int(n_c * 0.03)))
    i_hi = min(n_c - 1, max(0, int(n_c * 0.97)))

    min_co = Vector((xs[i_lo], ys[i_lo], zs[i_lo]))
    max_co = Vector((xs[i_hi], ys[i_hi], zs[i_hi]))
    center = (min_co + max_co) * 0.5
    half = (max_co - min_co) * 0.5

    rx = max(half.x, 1e-4)
    ry = max(half.y, 1e-4)
    rz = max(half.z * 1.25, 1e-4)

    out = dict(face_vert_normals)
    for v_idx in target_verts:
        d = mesh.vertices[v_idx].co - center
        ellip = Vector((d.x / rx, d.y / ry, d.z / rz))
        if ellip.length_squared <= 1e-12:
            continue
        n_sphere = ellip.normalized()
        n_proxy = face_vert_normals[v_idx]

        height_fade = 1.0
        if head_z is not None:
            height_fade = max(
                0.0, min(1.0, (mesh.vertices[v_idx].co.z - (head_z - 0.02)) / 0.05)
            )
        dot_pos = max(0.0, n_proxy.dot(n_sphere))
        local_w = blend * height_fade * (dot_pos**0.8)

        n_final = n_proxy.lerp(n_sphere, local_w)
        if n_final.length_squared > 1e-12:
            out[v_idx] = n_final.normalized()

    return out


def heal_same_material_seam_normals(mesh, loop_normals, cos_thresh=0.6428):
    """对同材质内部、间距 <=0.15mm 且几何折角 <50° 的未合并裂口顶点做一次法线平均，保留硬边同时消除改模碎影"""
    slot_verts = defaultdict(set)
    slot_loops_by_v = defaultdict(lambda: defaultdict(list))
    loops = mesh.loops

    for poly in mesh.polygons:
        s = poly.material_index
        for l_idx in poly.loop_indices:
            v_idx = loops[l_idx].vertex_index
            slot_verts[s].add(v_idx)
            slot_loops_by_v[s][v_idx].append(l_idx)

    healed_verts = 0
    for s, v_set in slot_verts.items():
        if len(v_set) < 2:
            continue
        v_list = list(v_set)
        v_norm = {}
        for v_idx in v_list:
            acc = Vector((0.0, 0.0, 0.0))
            for l_idx in slot_loops_by_v[s][v_idx]:
                acc += Vector(loop_normals[l_idx])
            if acc.length_squared > 1e-12:
                v_norm[v_idx] = acc.normalized()

        kd = kdtree.KDTree(len(v_list))
        for idx, v_idx in enumerate(v_list):
            kd.insert(mesh.vertices[v_idx].co, idx)
        kd.balance()

        for v_idx in v_list:
            n_base = v_norm.get(v_idx)
            if n_base is None:
                continue
            hits = kd.find_range(mesh.vertices[v_idx].co, 0.00015)
            if len(hits) <= 1:
                continue
            merged = n_base.copy()
            cnt = 1
            for _, o_idx, _ in hits:
                ov = v_list[o_idx]
                if ov == v_idx:
                    continue
                n_o = v_norm.get(ov)
                if n_o is not None and n_base.dot(n_o) >= cos_thresh:
                    merged += n_o
                    cnt += 1
            if cnt > 1 and merged.length_squared > 1e-12:
                avg_tup = tuple(merged.normalized())
                for l_idx in slot_loops_by_v[s][v_idx]:
                    loop_normals[l_idx] = avg_tup
                healed_verts += 1

    return healed_verts


def inspect_and_fix_mesh_anomalies(mesh, loop_normals):
    """修复全网格孤立反转循环法线（0<flip<35%）并检测同材质槽内部的重合面与硬平面异常"""
    loops = mesh.loops
    slot_polys = defaultdict(list)
    slot_vert_acc = defaultdict(lambda: defaultdict(Vector))

    for p_idx, poly in enumerate(mesh.polygons):
        s = poly.material_index
        slot_polys[s].append(p_idx)
        p_no = poly.normal
        if all(math.isfinite(x) for x in p_no) and p_no.length_squared > 1e-12:
            area_n = p_no * max(poly.area, 1e-8)
            for v_idx in poly.vertices:
                slot_vert_acc[s][v_idx] += area_n

    fixed_isolated = 0
    anomaly_notes = []

    for s, p_indices in sorted(slot_polys.items()):
        mat_name = (
            mesh.materials[s].name
            if 0 <= s < len(mesh.materials) and mesh.materials[s]
            else f"slot_{s}"
        )
        total_loops = 0
        flipped_list = []
        flat_faces = 0
        seen_centers = {}
        same_slot_dups = 0

        for p_idx in p_indices:
            poly = mesh.polygons[p_idx]
            p_no = poly.normal
            if not poly.use_smooth:
                flat_faces += 1
            c = poly.center
            ckey = (round(c.x, 4), round(c.y, 4), round(c.z, 4))
            if ckey in seen_centers:
                same_slot_dups += 1
            else:
                seen_centers[ckey] = p_idx

            for l_idx in poly.loop_indices:
                total_loops += 1
                ln = Vector(loop_normals[l_idx])
                if (
                    not all(math.isfinite(x) for x in ln)
                    or ln.length_squared < 1e-12
                    or (p_no.length_squared > 1e-12 and p_no.dot(ln) < 0.0)
                ):
                    flipped_list.append((p_idx, l_idx))

        flip_ratio = len(flipped_list) / max(1, total_loops)
        # 仅修复孤立反转循环法线（跳过 100% 反面的 目 / 裙内）
        if flipped_list and flip_ratio < 0.35:
            for p_idx, l_idx in flipped_list:
                poly = mesh.polygons[p_idx]
                v_idx = loops[l_idx].vertex_index
                vn = slot_vert_acc[s].get(v_idx)
                if (
                    vn is not None
                    and vn.length_squared > 1e-12
                    and poly.normal.dot(vn) > 0.0
                ):
                    loop_normals[l_idx] = tuple(vn.normalized())
                elif poly.normal.length_squared > 1e-12:
                    loop_normals[l_idx] = tuple(poly.normal.normalized())
                fixed_isolated += 1

        if same_slot_dups >= 10 or flat_faces >= 10:
            anomaly_notes.append(
                f"{mat_name}(同槽重合面={same_slot_dups}, 平面着色={flat_faces}, 孤立反法线={len(flipped_list)})"
            )

    if fixed_isolated:
        print(f"   🔧 [{mesh.name}] 已自动修复 {fixed_isolated} 个孤立反转/零法线循环")
    if anomaly_notes:
        print(f"   🔎 [{mesh.name}] 改模拓扑体检: {', '.join(anomaly_notes)}")


def repair_skin_normals(obj):
    mesh = obj.data
    mesh.update()
    if not mesh.polygons:
        return

    if hasattr(mesh, "corner_normals"):
        loop_normals = [tuple(cn.vector) for cn in mesh.corner_normals]
    else:
        mesh.calc_normals_split()
        loop_normals = [tuple(l.normal) for l in mesh.loops]

    slot_kinds = []
    for mat in mesh.materials:
        if not mat:
            slot_kinds.append("other")
            continue
        b_img = extract_material_base_image(mat)
        slot_kinds.append(
            classify_surface_for_normals(
                obj.name, mat.name, b_img.name if b_img else ""
            )
        )

    groups = {"face_skin": [], "body_skin": [], "body_mixed": []}
    for p_idx, poly in enumerate(mesh.polygons):
        kind = (
            slot_kinds[poly.material_index]
            if 0 <= poly.material_index < len(slot_kinds)
            else "other"
        )
        if kind in groups:
            groups[kind].append(p_idx)

    loops = mesh.loops
    cos_flip_guard = math.cos(math.radians(82.0))
    total_welded = 0

    def _safe_write_group(poly_list, vn_map, fallback_map, min_dot):
        for p_idx in poly_list:
            poly = mesh.polygons[p_idx]
            p_no = poly.normal
            if p_no.length_squared < 1e-12:
                continue
            cand = []
            valid = True
            for l_idx in poly.loop_indices:
                v_idx = loops[l_idx].vertex_index
                vn = vn_map.get(v_idx)
                if (
                    vn is None
                    or not all(math.isfinite(x) for x in vn)
                    or vn.length_squared < 1e-12
                    or p_no.dot(vn) < min_dot
                ):
                    valid = False
                    break
                cand.append((l_idx, tuple(vn.normalized())))
            if valid:
                poly.use_smooth = True
                for l_idx, n_tup in cand:
                    loop_normals[l_idx] = n_tup
            else:
                poly.use_smooth = True
                for l_idx in poly.loop_indices:
                    v_idx = loops[l_idx].vertex_index
                    fb = fallback_map.get(v_idx)
                    if (
                        fb is not None
                        and all(math.isfinite(x) for x in fb)
                        and fb.length_squared > 1e-12
                        and p_no.dot(fb) > 0.05
                    ):
                        loop_normals[l_idx] = tuple(fb.normalized())

    if groups["face_skin"]:
        face_vn, face_fb, w_cnt = compute_welded_smooth_normals(
            obj, groups["face_skin"], smooth_passes=4, crease_deg=75.0
        )
        total_welded += w_cnt
        face_vn = apply_face_ellipsoid_blend(obj, face_vn, blend=0.50)
        _safe_write_group(groups["face_skin"], face_vn, face_fb, cos_flip_guard)

    if groups["body_skin"]:
        skin_vn, skin_fb, w_cnt = compute_welded_smooth_normals(
            obj, groups["body_skin"], smooth_passes=4, crease_deg=70.0
        )
        total_welded += w_cnt
        _safe_write_group(groups["body_skin"], skin_vn, skin_fb, cos_flip_guard)

    if groups["body_mixed"]:
        mixed_vn, mixed_fb, w_cnt = compute_welded_smooth_normals(
            obj, groups["body_mixed"], smooth_passes=2, crease_deg=52.0
        )
        total_welded += w_cnt
        cos_crease = math.cos(math.radians(55.0))
        _safe_write_group(groups["body_mixed"], mixed_vn, mixed_fb, cos_crease)

    healed_cnt = heal_same_material_seam_normals(mesh, loop_normals)
    inspect_and_fix_mesh_anomalies(mesh, loop_normals)

    if hasattr(mesh, "use_auto_smooth"):
        mesh.use_auto_smooth = True
    mesh.normals_split_custom_set(loop_normals)
    mesh.update()
    print(
        f"   ✨ [{obj.name}] 法线重构完成: "
        f"脸部={len(groups['face_skin'])}面, 纯皮肤={len(groups['body_skin'])}面, "
        f"身体混合={len(groups['body_mixed'])}面 | 焊接代理裂缝={total_welded}点, 抚平同材质裂口={healed_cnt}点"
    )


def separate_coplanar_material_overlays(obj, offset=0.00025):
    """检测同一网格上完全重合的材质面，按叠加层数递增微推(+0.25mm, +0.5mm...)彻底消除 Z-fighting 与 EEVEE 光追自阴影变黑"""
    mesh = obj.data
    if not mesh.polygons or len(mesh.materials) < 2:
        return

    if hasattr(mesh, "corner_normals"):
        exact_normals = [tuple(cn.vector) for cn in mesh.corner_normals]
    else:
        mesh.calc_normals_split()
        exact_normals = [tuple(l.normal) for l in mesh.loops]

    slot_centers = defaultdict(set)
    slot_verts = defaultdict(set)
    slot_vert_dirs = defaultdict(lambda: defaultdict(Vector))

    for poly in mesh.polygons:
        s_idx = poly.material_index
        c = poly.center
        slot_centers[s_idx].add((round(c.x, 4), round(c.y, 4), round(c.z, 4)))
        p_no = poly.normal
        valid_no = all(math.isfinite(x) for x in p_no) and p_no.length_squared > 1e-12
        for v_idx in poly.vertices:
            slot_verts[s_idx].add(v_idx)
            if valid_no:
                slot_vert_dirs[s_idx][v_idx] += p_no

    slots = sorted(slot_centers.keys())
    pushed_slots = {}
    layer_count = defaultdict(int)
    for b_pos, s_b in enumerate(slots):
        c_b = slot_centers[s_b]
        if len(c_b) < 20:
            continue
        for s_a in slots[:b_pos]:
            c_a = slot_centers[s_a]
            if len(c_a) < 20:
                continue
            overlap = len(c_a & c_b)
            if overlap >= 20 and overlap >= 0.5 * min(len(c_a), len(c_b)):
                root_s = pushed_slots[s_a][0] if s_a in pushed_slots else s_a
                layer_count[root_s] += 1
                pushed_slots[s_b] = (root_s, layer_count[root_s])
                break

    if not pushed_slots:
        return

    vert_deltas = {}
    separated_names = []
    for s_b, (s_root, layer_idx) in pushed_slots.items():
        exclusive_verts = set(slot_verts[s_b]) - set(slot_verts[s_root])
        for other_b, (other_root, other_layer) in pushed_slots.items():
            if other_root == s_root and other_layer < layer_idx:
                exclusive_verts -= slot_verts[other_b]
        if not exclusive_verts:
            continue
        dirs_b = slot_vert_dirs[s_b]
        step_offset = offset * layer_idx
        cnt = 0
        for v_idx in exclusive_verts:
            if v_idx in vert_deltas:
                continue
            d = dirs_b.get(v_idx, Vector((0.0, 0.0, 0.0)))
            if d.length_squared > 1e-12:
                vert_deltas[v_idx] = d.normalized() * step_offset
                cnt += 1
        if cnt:
            mat_a = (
                mesh.materials[s_root].name if mesh.materials[s_root] else str(s_root)
            )
            mat_b = mesh.materials[s_b].name if mesh.materials[s_b] else str(s_b)
            separated_names.append(f"{mat_b}(覆盖{mat_a} L{layer_idx}, {cnt}点)")

    if not vert_deltas:
        return

    for v_idx, delta in vert_deltas.items():
        mesh.vertices[v_idx].co += delta

    if mesh.shape_keys and len(mesh.shape_keys.key_blocks) > 0:
        for kb in mesh.shape_keys.key_blocks:
            for v_idx, delta in vert_deltas.items():
                kb.data[v_idx].co += delta

    mesh.update()
    if hasattr(mesh, "use_auto_smooth"):
        mesh.use_auto_smooth = True
    mesh.normals_split_custom_set(exact_normals)
    mesh.update()
    print(f"   🛡️ [{obj.name}] 已按层级分离重合材质层: {', '.join(separated_names)}")


def push_vertices_along_normals(obj, strength=0.0004):
    mesh = obj.data
    if hasattr(mesh, "corner_normals"):
        exact_normals = [tuple(cn.vector) for cn in mesh.corner_normals]
    else:
        mesh.calc_normals_split()
        exact_normals = [tuple(l.normal) for l in mesh.loops]

    v_count = len(mesh.vertices)
    vert_dirs = [Vector((0.0, 0.0, 0.0)) for _ in range(v_count)]
    for l_idx, loop in enumerate(mesh.loops):
        vert_dirs[loop.vertex_index] += Vector(exact_normals[l_idx])

    for i in range(v_count):
        if vert_dirs[i].length_squared > 1e-12:
            vert_dirs[i].normalize()
        else:
            vert_dirs[i] = Vector(mesh.vertices[i].normal)

    if mesh.shape_keys and len(mesh.shape_keys.key_blocks) > 0:
        for kb in mesh.shape_keys.key_blocks:
            for i, kb_vert in enumerate(kb.data):
                kb_vert.co += vert_dirs[i] * strength
        mesh.update()
    else:
        for i, vert in enumerate(mesh.vertices):
            vert.co += vert_dirs[i] * strength
        mesh.update()

    if hasattr(mesh, "use_auto_smooth"):
        mesh.use_auto_smooth = True
    mesh.normals_split_custom_set(exact_normals)
    mesh.update()


def is_mmd_physics_or_garbage(obj):
    if getattr(obj, "mmd_type", "NONE") in ("RIGID_BODY", "JOINT"):
        return True
    if obj.type == "EMPTY" and obj.name.lower() in ("joints", "rigidbodies"):
        return True
    if obj.parent and obj.parent.name.lower() in ("joints", "rigidbodies"):
        return True
    for col in obj.users_collection:
        if col.name.lower() in ("rigidbodies", "joints"):
            return True
    if obj.type == "MESH":
        name_low = obj.name.lower()
        if any(
            kw in name_low
            for kw in (
                "shadow",
                "dummy",
                "mmd_shadow",
                "mmd_dummy",
                "rigidbody",
                "mmd_rigid",
                "mmd_joint",
                "cs_rigid",
            )
        ):
            return True
    return False


def clear_mmd_armature_constraints():
    """解除 mmd_tools 生成的 _dummy_/_shadow_ 骨骼约束，防止步骤 0 依赖图刷新时刷屏循环依赖警告"""
    cleared = 0
    for arm in bpy.data.objects:
        if arm.type != "ARMATURE" or not arm.pose:
            continue
        for pb in arm.pose.bones:
            for c in list(pb.constraints):
                pb.constraints.remove(c)
                cleared += 1
            pb.location = (0.0, 0.0, 0.0)
            pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
            pb.rotation_euler = (0.0, 0.0, 0.0)
            pb.rotation_axis_angle = (0.0, 0.0, 1.0, 0.0)
            pb.scale = (1.0, 1.0, 1.0)
    if cleared:
        print(f"   🔓 已解除 {cleared} 条 MMD 骨骼循环约束")


def run():
    print(">>> [步骤 0] 无损法线保护 + 皮肤代理法线重构 + 清理 mmd_tools 刚体/关节...")

    snapshot_mmd_physics_to_armature()
    clear_mmd_armature_constraints()

    removed_cnt = 0
    for obj in list(bpy.data.objects):
        if is_mmd_physics_or_garbage(obj):
            bpy.data.objects.remove(obj, do_unlink=True)
            removed_cnt += 1
    if removed_cnt:
        print(f"   🧹 已清理 {removed_cnt} 个冗余刚体/关节/Dummy对象")

    for obj in list(bpy.data.objects):
        if obj.type != "MESH" or not obj.data:
            continue

        try:
            if not obj.data.has_custom_normals:
                for poly in obj.data.polygons:
                    poly.use_smooth = True
        except Exception:
            pass

        if obj.data.color_attributes:
            try:
                while obj.data.color_attributes:
                    obj.data.color_attributes.remove(obj.data.color_attributes[0])
            except Exception:
                pass

        has_shape_keys = bool(
            obj.data.shape_keys and len(obj.data.shape_keys.key_blocks) > 1
        )

        if not has_shape_keys:
            kd, orig_loop_normals = None, None
            try:
                kd, orig_loop_normals = snapshot_custom_normals(obj.data)
            except Exception:
                pass
            try:
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode="EDIT")
                bpy.ops.mesh.select_all(action="SELECT")
                bpy.ops.mesh.remove_doubles(threshold=0.00005)
                bpy.ops.mesh.dissolve_degenerate(threshold=0.0001)
                bpy.ops.object.mode_set(mode="OBJECT")
            except Exception as e:
                print(f"   ⚠️ 网格 [{obj.name}] 几何清理跳过: {e}")
                try:
                    bpy.ops.object.mode_set(mode="OBJECT")
                except Exception:
                    pass
            if kd is not None and orig_loop_normals:
                try:
                    restore_custom_normals(obj.data, kd, orig_loop_normals)
                except Exception:
                    pass

        try:
            bpy.context.view_layer.objects.active = obj
            repair_skin_normals(obj)
        except Exception as e:
            print(f"   ⚠️ 网格 [{obj.name}] 皮肤法线重构跳过: {e}")

        try:
            separate_coplanar_material_overlays(obj, offset=0.00025)
        except Exception as e:
            print(f"   ⚠️ 网格 [{obj.name}] 重合材质分离跳过: {e}")

        obj_name_lower = obj.name.lower()
        is_clothing = any(
            kw in obj_name_lower
            for kw in ["cloth", "wear", "dress", "costume", "jacket", "skirt"]
        )
        is_inner_skin = any(
            kw in obj_name_lower for kw in ["skin", "inner", "naked", "kao", "face"]
        )

        if is_clothing and not is_inner_skin:
            try:
                push_vertices_along_normals(obj, strength=0.0004)
            except Exception as e:
                print(f"   ⚠️ 衣服网格 [{obj.name}] 微推失败: {e}")

    print("✅ [步骤 0] 皮肤法线重构与清理完成！")
