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

pub(crate) fn size_and_align_of<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    layout: TyAndLayout<'tcx>,
    info: Option<Value>,
) -> (Value, Value) {
    if layout.is_sized() {
        return (
            fx.bcx.ins().iconst(fx.pointer_type, layout.size.bytes() as i64),
            fx.bcx.ins().iconst(fx.pointer_type, layout.align.bytes() as i64),
        );
    }

    let ty = layout.ty;
    match ty.kind() {
        ty::Dynamic(..) => {
            // load size/align from vtable
            (
                crate::vtable::size_of_obj(fx, info.unwrap()),
                crate::vtable::align_of_obj(fx, info.unwrap()),
            )
        }
        ty::Slice(_) | ty::Str => {
            let unit = layout.field(fx, 0);
            // The info in this case is the length of the str, so the size is that
            // times the unit size.
            (
                fx.bcx.ins().imul_imm(info.unwrap(), unit.size.bytes() as i64),
                fx.bcx.ins().iconst(fx.pointer_type, unit.align.bytes() as i64),
            )
        }
        ty::Foreign(_) => {
            let trap_block = fx.bcx.create_block();
            let true_ = fx.bcx.ins().iconst(types::I8, 1);
            let next_block = fx.bcx.create_block();
            fx.bcx.ins().brif(true_, trap_block, &[], next_block, &[]);
            fx.bcx.seal_block(trap_block);
            fx.bcx.seal_block(next_block);
            fx.bcx.switch_to_block(trap_block);

            // `extern` type. We cannot compute the size, so panic.
            let msg_str = with_no_visible_paths!({
                with_no_trimmed_paths!({
                    format!("attempted to compute the size or alignment of extern type `{ty}`")
                })
            });

            codegen_panic_nounwind(fx, &msg_str, fx.mir.span);

            fx.bcx.switch_to_block(next_block);

            // This function does not return so we can now return whatever we want.
            let size = fx.bcx.ins().iconst(fx.pointer_type, 42);
            let align = fx.bcx.ins().iconst(fx.pointer_type, 42);
            (size, align)
        }
        ty::Adt(..) | ty::Tuple(..) => {
            // First get the size of all statically known fields.
            // Don't use size_of because it also rounds up to alignment, which we
            // want to avoid, as the unsized field's alignment could be smaller.
            assert!(!layout.ty.is_simd());

            let i = layout.fields.count() - 1;
            let unsized_offset_unadjusted = layout.fields.offset(i).bytes();
            let unsized_offset_unadjusted =
                fx.bcx.ins().iconst(fx.pointer_type, unsized_offset_unadjusted as i64);
            let sized_align = layout.align.bytes();
            let sized_align = fx.bcx.ins().iconst(fx.pointer_type, sized_align as i64);

            // Recurse to get the size of the dynamically sized field (must be
            // the last field).
            let field_layout = layout.field(fx, i);
            let (unsized_size, mut unsized_align) = size_and_align_of(fx, field_layout, info);

            // # First compute the dynamic alignment

            // For packed types, we need to cap the alignment.
            if let ty::Adt(def, _) = ty.kind() {
                if let Some(packed) = def.repr().pack {
                    if packed.bytes() == 1 {
                        // We know this will be capped to 1.
                        unsized_align = fx.bcx.ins().iconst(fx.pointer_type, 1);
                    } else {
                        // We have to dynamically compute `min(unsized_align, packed)`.
                        let packed = fx.bcx.ins().iconst(fx.pointer_type, packed.bytes() as i64);
                        let cmp = fx.bcx.ins().icmp(IntCC::UnsignedLessThan, unsized_align, packed);
                        unsized_align = fx.bcx.ins().select(cmp, unsized_align, packed);
                    }
                }
            }

            // Choose max of two known alignments (combined value must
            // be aligned according to more restrictive of the two).
            let cmp = fx.bcx.ins().icmp(IntCC::UnsignedGreaterThan, sized_align, unsized_align);
            let full_align = fx.bcx.ins().select(cmp, sized_align, unsized_align);

            // # Then compute the dynamic size

            // The full formula for the size would be:
            // let unsized_offset_adjusted = unsized_offset_unadjusted.align_to(unsized_align);
            // let full_size = (unsized_offset_adjusted + unsized_size).align_to(full_align);
            // However, `unsized_size` is a multiple of `unsized_align`.
            // Therefore, we can equivalently do the `align_to(unsized_align)` *after* adding `unsized_size`:
            // let full_size = (unsized_offset_unadjusted + unsized_size).align_to(unsized_align).align_to(full_align);
            // Furthermore, `align >= unsized_align`, and therefore we only need to do:
            // let full_size = (unsized_offset_unadjusted + unsized_size).align_to(full_align);

            let full_size = fx.bcx.ins().iadd(unsized_offset_unadjusted, unsized_size);

            // Issue #27023: must add any necessary padding to `size`
            // (to make it a multiple of `align`) before returning it.
            //
            // Namely, the returned size should be, in C notation:
            //
            //   `size + ((size & (align-1)) ? align : 0)`
            //
            // emulated via the semi-standard fast bit trick:
            //
            //   `(size + (align-1)) & -align`
            let addend = fx.bcx.ins().iadd_imm(full_align, -1);
            let add = fx.bcx.ins().iadd(full_size, addend);
            let neg = fx.bcx.ins().ineg(full_align);
            let full_size = fx.bcx.ins().band(add, neg);

            (full_size, full_align)
        }
        _ => bug!("size_and_align_of_dst: {ty} not supported"),
    }
}
