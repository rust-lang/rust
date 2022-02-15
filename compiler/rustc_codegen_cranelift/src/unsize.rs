//! Codegen of the [`PointerCast::Unsize`] operation.
//!
//! [`PointerCast::Unsize`]: `rustc_middle::ty::adjustment::PointerCast::Unsize`

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
        fx.tcx.struct_lockstep_tails_erasing_lifetimes(source, target, ParamEnv::reveal_all());
    match (&source.kind(), &target.kind()) {
        (&ty::Array(_, len), &ty::Slice(_)) => fx
            .bcx
            .ins()
            .iconst(fx.pointer_type, len.eval_usize(fx.tcx, ParamEnv::reveal_all()) as i64),
        (&ty::Dynamic(ref data_a, ..), &ty::Dynamic(ref data_b, ..)) => {
            let old_info =
                old_info.expect("unsized_info: missing old info for trait upcasting coercion");
            if data_a.principal_def_id() == data_b.principal_def_id() {
                return old_info;
            }

            // trait upcasting coercion
            let vptr_entry_idx =
                fx.tcx.vtable_trait_upcasting_coercion_new_vptr_slot((source, target));

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
        (_, &ty::Dynamic(ref data, ..)) => crate::vtable::get_vtable(fx, source, data.principal()),
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
        | (&ty::Ref(_, a, _), &ty::RawPtr(ty::TypeAndMut { ty: b, .. }))
        | (&ty::RawPtr(ty::TypeAndMut { ty: a, .. }), &ty::RawPtr(ty::TypeAndMut { ty: b, .. })) => {
            (src, unsized_info(fx, *a, *b, old_info))
        }
        (&ty::Adt(def_a, _), &ty::Adt(def_b, _)) if def_a.is_box() && def_b.is_box() => {
            let (a, b) = (src_layout.ty.boxed_ty(), dst_layout.ty.boxed_ty());
            (src, unsized_info(fx, a, b, old_info))
        }
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
                if src_f.is_zst() {
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
            if fx.layout_of(src.layout().ty.builtin_deref(true).unwrap().ty).is_unsized() {
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

            for i in 0..def_a.variants[VariantIdx::new(0)].fields.len() {
                let src_f = src.value_field(fx, mir::Field::new(i));
                let dst_f = dst.place_field(fx, mir::Field::new(i));

                if dst_f.layout().is_zst() {
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

pub(crate) fn size_and_align_of_dst<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    layout: TyAndLayout<'tcx>,
    info: Value,
) -> (Value, Value) {
    if !layout.is_unsized() {
        let size = fx.bcx.ins().iconst(fx.pointer_type, layout.size.bytes() as i64);
        let align = fx.bcx.ins().iconst(fx.pointer_type, layout.align.abi.bytes() as i64);
        return (size, align);
    }
    match layout.ty.kind() {
        ty::Dynamic(..) => {
            // load size/align from vtable
            (crate::vtable::size_of_obj(fx, info), crate::vtable::min_align_of_obj(fx, info))
        }
        ty::Slice(_) | ty::Str => {
            let unit = layout.field(fx, 0);
            // The info in this case is the length of the str, so the size is that
            // times the unit size.
            (
                fx.bcx.ins().imul_imm(info, unit.size.bytes() as i64),
                fx.bcx.ins().iconst(fx.pointer_type, unit.align.abi.bytes() as i64),
            )
        }
        _ => {
            // First get the size of all statically known fields.
            // Don't use size_of because it also rounds up to alignment, which we
            // want to avoid, as the unsized field's alignment could be smaller.
            assert!(!layout.ty.is_simd());

            let i = layout.fields.count() - 1;
            let sized_size = layout.fields.offset(i).bytes();
            let sized_align = layout.align.abi.bytes();
            let sized_align = fx.bcx.ins().iconst(fx.pointer_type, sized_align as i64);

            // Recurse to get the size of the dynamically sized field (must be
            // the last field).
            let field_layout = layout.field(fx, i);
            let (unsized_size, mut unsized_align) = size_and_align_of_dst(fx, field_layout, info);

            // FIXME (#26403, #27023): We should be adding padding
            // to `sized_size` (to accommodate the `unsized_align`
            // required of the unsized field that follows) before
            // summing it with `sized_size`. (Note that since #26403
            // is unfixed, we do not yet add the necessary padding
            // here. But this is where the add would go.)

            // Return the sum of sizes and max of aligns.
            let size = fx.bcx.ins().iadd_imm(unsized_size, sized_size as i64);

            // Packed types ignore the alignment of their fields.
            if let ty::Adt(def, _) = layout.ty.kind() {
                if def.repr.packed() {
                    unsized_align = sized_align;
                }
            }

            // Choose max of two known alignments (combined value must
            // be aligned according to more restrictive of the two).
            let cmp = fx.bcx.ins().icmp(IntCC::UnsignedGreaterThan, sized_align, unsized_align);
            let align = fx.bcx.ins().select(cmp, sized_align, unsized_align);

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
            let addend = fx.bcx.ins().iadd_imm(align, -1);
            let add = fx.bcx.ins().iadd(size, addend);
            let neg = fx.bcx.ins().ineg(align);
            let size = fx.bcx.ins().band(add, neg);

            (size, align)
        }
    }
}
