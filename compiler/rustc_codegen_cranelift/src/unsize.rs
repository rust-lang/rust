//! Codegen of the [`PointerCoercion::Unsize`] operation.
//!
//! [`PointerCoercion::Unsize`]: `rustc_middle::ty::adjustment::PointerCoercion::Unsize`

use rustc_codegen_ssa::base::validate_trivial_unsize;
use rustc_middle::ty::layout::HasTypingEnv;
use rustc_middle::ty::print::{with_no_trimmed_paths, with_no_visible_paths};

use crate::base::codegen_panic_nounwind;
use crate::prelude::*;

// Adapted from https://github.com/rust-lang/rust/blob/2a663555ddf36f6b041445894a8c175cd1bc718c/src/librustc_codegen_ssa/base.rs#L159-L307

/// Retrieve the information we are losing (making dynamic) in an unsizing
/// adjustment.
///
/// The `old_info` argument is a bit funny. It is intended for use
/// in an upcast, where the new vtable for an object will be derived
/// from the old one.
pub(crate) fn unsized_info<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    source: Ty<'tcx>,
    target: Ty<'tcx>,
    old_info: Option<Value>,
) -> Value {
    let (source, target) =
        fx.tcx.struct_lockstep_tails_for_codegen(source, target, fx.typing_env());
    match (&source.kind(), &target.kind()) {
        (&ty::Array(_, len), &ty::Slice(_)) => fx.bcx.ins().iconst(
            fx.pointer_type,
            len.try_to_target_usize(fx.tcx).expect("expected monomorphic const in codegen") as i64,
        ),
        (&ty::Dynamic(data_a, _), &ty::Dynamic(data_b, _)) => {
            let old_info =
                old_info.expect("unsized_info: missing old info for trait upcasting coercion");
            let b_principal_def_id = data_b.principal_def_id();
            if data_a.principal_def_id() == b_principal_def_id || b_principal_def_id.is_none() {
                // A NOP cast that doesn't actually change anything, should be allowed even with invalid vtables.
                debug_assert!(
                    validate_trivial_unsize(fx.tcx, data_a, data_b),
                    "NOP unsize vtable changed principal trait ref: {data_a} -> {data_b}"
                );
                return old_info;
            }

            // trait upcasting coercion
            let vptr_entry_idx = fx.tcx.supertrait_vtable_slot((source, target));

            if let Some(entry_idx) = vptr_entry_idx {
                let entry_idx = u32::try_from(entry_idx).unwrap();
                let entry_offset = entry_idx * fx.pointer_type.bytes();
                let vptr_ptr = Pointer::new(old_info).offset_i64(fx, entry_offset.into()).load(
                    fx,
                    fx.pointer_type,
                    crate::vtable::vtable_memflags(),
                );
                vptr_ptr
            } else {
                old_info
            }
        }
        (_, ty::Dynamic(data, ..)) => crate::vtable::get_vtable(
            fx,
            source,
            data.principal()
                .map(|principal| fx.tcx.instantiate_bound_regions_with_erased(principal)),
        ),
        _ => bug!("unsized_info: invalid unsizing {:?} -> {:?}", source, target),
    }
}

/// Coerce `src` to `dst_ty`.
fn unsize_ptr<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    src: Value,
    src_layout: TyAndLayout<'tcx>,
    dst_layout: TyAndLayout<'tcx>,
    old_info: Option<Value>,
) -> (Value, Value) {
    match (&src_layout.ty.kind(), &dst_layout.ty.kind()) {
        (&ty::Ref(_, a, _), &ty::Ref(_, b, _))
        | (&ty::Ref(_, a, _), &ty::RawPtr(b, _))
        | (&ty::RawPtr(a, _), &ty::RawPtr(b, _)) => (src, unsized_info(fx, *a, *b, old_info)),
        (&ty::Adt(def_a, _), &ty::Adt(def_b, _)) => {
            assert_eq!(def_a, def_b);

            if src_layout == dst_layout {
                return (src, old_info.unwrap());
            }

            let mut result = None;
            for i in 0..src_layout.fields.count() {
                let src_f = src_layout.field(fx, i);
                assert_eq!(src_layout.fields.offset(i).bytes(), 0);
                assert_eq!(dst_layout.fields.offset(i).bytes(), 0);
                if src_f.is_1zst() {
                    // We are looking for the one non-1-ZST field; this is not it.
                    continue;
                }
                assert_eq!(src_layout.size, src_f.size);

                let dst_f = dst_layout.field(fx, i);
                assert_ne!(src_f.ty, dst_f.ty);
                assert_eq!(result, None);
                result = Some(unsize_ptr(fx, src, src_f, dst_f, old_info));
            }
            result.unwrap()
        }
        _ => bug!("unsize_ptr: called on bad types"),
    }
}

/// Coerce `src`, which is a reference to a value of type `src_ty`,
/// to a value of type `dst_ty` and store the result in `dst`
pub(crate) fn coerce_unsized_into<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    src: CValue<'tcx>,
    dst: CPlace<'tcx>,
) {
    let src_ty = src.layout().ty;
    let dst_ty = dst.layout().ty;
    let mut coerce_ptr = || {
        let (base, info) =
            if fx.layout_of(src.layout().ty.builtin_deref(true).unwrap()).is_unsized() {
                let (old_base, old_info) = src.load_scalar_pair(fx);
                unsize_ptr(fx, old_base, src.layout(), dst.layout(), Some(old_info))
            } else {
                let base = src.load_scalar(fx);
                unsize_ptr(fx, base, src.layout(), dst.layout(), None)
            };
        dst.write_cvalue(fx, CValue::by_val_pair(base, info, dst.layout()));
    };
    match (&src_ty.kind(), &dst_ty.kind()) {
        (ty::Pat(a, _), ty::Pat(b, _)) => {
            let src = src.cast_pat_ty_to_base(fx.layout_of(*a));
            let dst = dst.place_transmute_type(fx, *b);
            coerce_unsized_into(fx, src, dst)
        }
        (&ty::Ref(..), &ty::Ref(..))
        | (&ty::Ref(..), &ty::RawPtr(..))
        | (&ty::RawPtr(..), &ty::RawPtr(..)) => coerce_ptr(),
        (&ty::Adt(def_a, _), &ty::Adt(def_b, _)) => {
            assert_eq!(def_a, def_b);

            for i in 0..def_a.variant(FIRST_VARIANT).fields.len() {
                let src_f = src.value_field(fx, FieldIdx::new(i));
                let dst_f = dst.place_field(fx, FieldIdx::new(i));

                if dst_f.layout().is_zst() {
                    // No data here, nothing to copy/coerce.
                    continue;
                }

                if src_f.layout().ty == dst_f.layout().ty {
                    dst_f.write_cvalue(fx, src_f);
                } else {
                    coerce_unsized_into(fx, src_f, dst_f);
                }
            }
        }
        _ => bug!("coerce_unsized_into: invalid coercion {:?} -> {:?}", src_ty, dst_ty),
    }
}

// Adapted from https://github.com/rust-lang/rust/blob/2a663555ddf36f6b041445894a8c175cd1bc718c/src/librustc_codegen_ssa/glue.rs

use rustc_codegen_ssa::size_of_val::unsized_tail_and_min_pack;

fn codegen_foreign_panic<'tcx>(fx: &mut FunctionCx<'_, '_, 'tcx>, ty: Ty<'tcx>) -> Value {
    let trap_block = fx.bcx.create_block();
    let true_ = fx.bcx.ins().iconst(types::I8, 1);
    let next_block = fx.bcx.create_block();
    fx.bcx.ins().brif(true_, trap_block, &[], next_block, &[]);
    fx.bcx.seal_block(trap_block);
    fx.bcx.seal_block(next_block);
    fx.bcx.switch_to_block(trap_block);

    // `extern` type. We cannot compute the size or alignment, so panic.
    let msg_str = with_no_visible_paths!({
        with_no_trimmed_paths!({
            format!("attempted to compute the size or alignment of extern type `{ty}`")
        })
    });

    codegen_panic_nounwind(fx, &msg_str, fx.mir.span);

    fx.bcx.switch_to_block(next_block);

    // This function does not return so we can now return whatever we want.
    fx.bcx.ins().iconst(fx.pointer_type, 42)
}

fn clamp_dyn_align(
    fx: &mut FunctionCx<'_, '_, '_>,
    mut dyn_align: Value,
    static_align: u64,
    min_pack: u64,
) -> Value {
    if min_pack < u64::MAX {
        let packed = fx.bcx.ins().iconst(fx.pointer_type, min_pack as i64);
        let cmp = fx.bcx.ins().icmp(IntCC::UnsignedLessThan, dyn_align, packed);
        dyn_align = fx.bcx.ins().select(cmp, dyn_align, packed);
    }
    if static_align > 1 {
        let sized_align = fx.bcx.ins().iconst(fx.pointer_type, static_align as i64);
        let cmp = fx.bcx.ins().icmp(IntCC::UnsignedGreaterThan, sized_align, dyn_align);
        dyn_align = fx.bcx.ins().select(cmp, sized_align, dyn_align);
    }
    dyn_align
}

pub(crate) fn dst_field_offset<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    parent: TyAndLayout<'tcx>,
    field: TyAndLayout<'tcx>,
    offset: Size,
    meta: Option<Value>,
) -> Option<Value> {
    if field.is_sized() || offset.bytes() == 0 {
        return None;
    }

    let (tail_ty, min_pack) = unsized_tail_and_min_pack(fx.tcx, fx.typing_env(), parent.ty);
    match tail_ty.kind() {
        ty::Slice(_) | ty::Str => None,
        ty::Dynamic(..) if offset.bytes().is_multiple_of(min_pack) => None,
        ty::Dynamic(..) => {
            let vtable_align = crate::vtable::align_of_obj(fx, meta.unwrap());
            let field_static_align = field.align.bytes();
            let offset_bytes = offset.bytes();
            debug_assert!(offset_bytes.is_multiple_of(field_static_align));
            debug_assert!(field_static_align < min_pack);
            if offset_bytes == field_static_align {
                Some(clamp_dyn_align(fx, vtable_align, field_static_align, min_pack))
            } else {
                let align = clamp_dyn_align(fx, vtable_align, 1, min_pack);
                let align_minus_1 = fx.bcx.ins().iadd_imm_s(align, -1);
                let mask = fx.bcx.ins().iconst(fx.pointer_type, offset_bytes.wrapping_neg() as i64);
                let pad = fx.bcx.ins().band(align_minus_1, mask);
                Some(fx.bcx.ins().iadd_imm_u(pad, offset_bytes as i64))
            }
        }
        ty::Foreign(_) => Some(codegen_foreign_panic(fx, tail_ty)),
        _ => bug!("dst_field_offset: unexpected unsized tail {tail_ty:?}"),
    }
}

pub(crate) fn align_of_dst<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    layout: TyAndLayout<'tcx>,
    info: Option<Value>,
) -> Value {
    if layout.is_sized() {
        return fx.bcx.ins().iconst(fx.pointer_type, layout.align.bytes() as i64);
    }

    let (tail_ty, min_pack) = unsized_tail_and_min_pack(fx.tcx, fx.typing_env(), layout.ty);
    match tail_ty.kind() {
        ty::Slice(_) | ty::Str => fx.bcx.ins().iconst(fx.pointer_type, layout.align.bytes() as i64),
        ty::Foreign(_) => codegen_foreign_panic(fx, tail_ty),
        ty::Dynamic(..) => {
            let static_align = layout.align.bytes();
            if min_pack <= static_align {
                return fx.bcx.ins().iconst(fx.pointer_type, static_align as i64);
            }
            let dyn_align = crate::vtable::align_of_obj(fx, info.unwrap());
            clamp_dyn_align(fx, dyn_align, static_align, min_pack)
        }
        _ => bug!("align_of_dst: {} (tail {tail_ty}) not supported", layout.ty),
    }
}

#[derive(Copy, Clone)]
struct DstChainStep {
    offset: u64,
    align: u64,
    pack: u64,
}

fn emit_aligned_static_size(
    fx: &mut FunctionCx<'_, '_, '_>,
    base: Value,
    base_multiple: u64,
    offset: u64,
    align: u64,
) -> Value {
    let (addend, mask_align) = if base_multiple.is_multiple_of(align) {
        (offset.next_multiple_of(align), 1)
    } else {
        (offset + (align - 1), align)
    };
    let sum = if addend == 0 { base } else { fx.bcx.ins().iadd_imm_u(base, addend as i64) };
    if mask_align == 1 {
        sum
    } else {
        let mask = fx.bcx.ins().iconst(fx.pointer_type, mask_align.wrapping_neg() as i64);
        fx.bcx.ins().band(sum, mask)
    }
}

fn emit_dyn_step(
    fx: &mut FunctionCx<'_, '_, '_>,
    cur_size: Value,
    base_multiple: Option<u64>,
    step: DstChainStep,
    vtable_align: &mut Option<Value>,
    vtable: Value,
) -> Value {
    if step.pack <= step.align {
        return emit_aligned_static_size(
            fx,
            cur_size,
            base_multiple.unwrap_or(1),
            step.offset,
            step.align,
        );
    }

    let raw_dyn_align =
        *vtable_align.get_or_insert_with(|| crate::vtable::align_of_obj(fx, vtable));
    let full_align = clamp_dyn_align(fx, raw_dyn_align, step.align, step.pack);

    if base_multiple.is_none() && step.offset == 1 && step.align == 1 {
        return fx.bcx.ins().iadd(cur_size, full_align);
    }

    let sum = if step.offset == 0 {
        let addend = fx.bcx.ins().iadd_imm_s(full_align, -1);
        fx.bcx.ins().iadd(cur_size, addend)
    } else if step.offset == 1 {
        fx.bcx.ins().iadd(cur_size, full_align)
    } else {
        let s = fx.bcx.ins().iadd_imm_u(cur_size, (step.offset - 1) as i64);
        fx.bcx.ins().iadd(s, full_align)
    };
    let neg = fx.bcx.ins().ineg(full_align);
    fx.bcx.ins().band(sum, neg)
}

pub(crate) fn size_of_dst<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    layout: TyAndLayout<'tcx>,
    info: Option<Value>,
) -> Value {
    if layout.is_sized() {
        return fx.bcx.ins().iconst(fx.pointer_type, layout.size.bytes() as i64);
    }

    let mut steps = Vec::new();
    let mut cur_layout = layout;
    loop {
        let cur_ty = cur_layout.ty;
        match cur_ty.kind() {
            ty::Adt(..) | ty::Tuple(..) => {
                assert!(!cur_ty.is_simd());
                let i = cur_layout.fields.count() - 1;
                let offset = cur_layout.fields.offset(i).bytes();
                let align = cur_layout.align.bytes();
                let pack = match cur_ty.kind() {
                    ty::Adt(def, _) => def.repr().pack.map_or(u64::MAX, |p| p.bytes()),
                    _ => u64::MAX,
                };
                steps.push(DstChainStep { offset, align, pack });
                cur_layout = cur_layout.field(fx, i);
            }
            ty::Pat(..) => {
                cur_layout = cur_layout.field(fx, 0);
            }
            ty::Slice(_) | ty::Str => {
                let unit = cur_layout.field(fx, 0);
                let elem_size = unit.size.bytes();

                if elem_size == 0 {
                    let size = steps
                        .iter()
                        .rev()
                        .fold(0u64, |size, step| (step.offset + size).next_multiple_of(step.align));
                    return fx.bcx.ins().iconst(fx.pointer_type, size as i64);
                }

                let len = info.unwrap();
                let base = if elem_size == 1 {
                    len
                } else {
                    fx.bcx.ins().imul_imm_u(len, elem_size as i64)
                };

                let mut offset = 0u64;
                let mut outer_offset = 0u64;
                let mut align = 1u64;

                for step in steps.iter().rev() {
                    if align <= step.align {
                        offset += (outer_offset + step.offset).next_multiple_of(align);
                        outer_offset = 0;
                        align = step.align;
                        if elem_size.is_multiple_of(align) {
                            offset = offset.next_multiple_of(align);
                            align = 1;
                        }
                    } else {
                        let c = (outer_offset + step.offset).next_multiple_of(step.align);
                        offset += c & !(align - 1);
                        outer_offset = c & (align - 1);
                    }
                }

                let aligned = emit_aligned_static_size(fx, base, elem_size, offset, align);
                return if outer_offset == 0 {
                    aligned
                } else {
                    fx.bcx.ins().iadd_imm_u(aligned, outer_offset as i64)
                };
            }
            ty::Dynamic(..) => {
                let vtable = info.unwrap();
                let mut cur_size = crate::vtable::size_of_obj(fx, vtable);
                let mut vtable_align: Option<Value> = None;
                let mut base_multiple: Option<u64> = None;
                let mut min_pack = u64::MAX;
                let mut pending: Option<DstChainStep> = None;

                for step in steps.iter().rev() {
                    min_pack = min_pack.min(step.pack);
                    let cur =
                        DstChainStep { offset: step.offset, align: step.align, pack: min_pack };

                    let Some(prev) = &mut pending else {
                        if cur.offset == 0 && cur.align == 1 {
                            continue;
                        }
                        pending = Some(cur);
                        continue;
                    };

                    if prev.align <= cur.align
                        && ((cur.offset == 0 && cur.pack == prev.pack)
                            || (prev.pack <= prev.align && cur.pack <= cur.align))
                    {
                        debug_assert!(cur.offset.is_multiple_of(prev.align));
                        prev.offset += cur.offset;
                        prev.align = cur.align;
                        prev.pack = cur.pack;
                        continue;
                    }

                    cur_size = emit_dyn_step(
                        fx,
                        cur_size,
                        base_multiple,
                        *prev,
                        &mut vtable_align,
                        vtable,
                    );
                    base_multiple = Some(prev.align);
                    *prev = cur;
                }

                if let Some(last) = pending {
                    cur_size =
                        emit_dyn_step(fx, cur_size, base_multiple, last, &mut vtable_align, vtable);
                }

                return cur_size;
            }
            ty::Foreign(_) => return codegen_foreign_panic(fx, cur_ty),
            _ => bug!("size_of_dst: {} (tail {cur_ty}) not supported", layout.ty),
        }
    }
}
