//! Codegen vtables and vtable accesses.
//!
//! See `rustc_codegen_ssa/src/meth.rs` for reference.

use crate::constant::data_id_for_vtable;
use crate::prelude::*;

pub(crate) fn vtable_memflags() -> MemFlags {
    let mut flags = MemFlags::trusted(); // A vtable access is always aligned and will never trap.
    flags.set_readonly(); // A vtable is always read-only.
    flags
}

pub(crate) fn drop_fn_of_obj(fx: &mut FunctionCx<'_, '_, '_>, vtable: Value) -> Value {
    let usize_size = fx.layout_of(fx.tcx.types.usize).size.bytes() as usize;
    fx.bcx.ins().load(
        fx.pointer_type,
        vtable_memflags(),
        vtable,
        (ty::COMMON_VTABLE_ENTRIES_DROPINPLACE * usize_size) as i32,
    )
}

pub(crate) fn size_of_obj(fx: &mut FunctionCx<'_, '_, '_>, vtable: Value) -> Value {
    let usize_size = fx.layout_of(fx.tcx.types.usize).size.bytes() as usize;
    fx.bcx.ins().load(
        fx.pointer_type,
        vtable_memflags(),
        vtable,
        (ty::COMMON_VTABLE_ENTRIES_SIZE * usize_size) as i32,
    )
}

pub(crate) fn min_align_of_obj(fx: &mut FunctionCx<'_, '_, '_>, vtable: Value) -> Value {
    let usize_size = fx.layout_of(fx.tcx.types.usize).size.bytes() as usize;
    fx.bcx.ins().load(
        fx.pointer_type,
        vtable_memflags(),
        vtable,
        (ty::COMMON_VTABLE_ENTRIES_ALIGN * usize_size) as i32,
    )
}

pub(crate) fn get_ptr_and_method_ref<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    mut arg: CValue<'tcx>,
    idx: usize,
) -> (Pointer, Value) {
    let (ptr, vtable) = 'block: {
        if let BackendRepr::Scalar(_) = arg.layout().backend_repr {
            while !arg.layout().ty.is_raw_ptr() && !arg.layout().ty.is_ref() {
                let (idx, _) = arg
                    .layout()
                    .non_1zst_field(fx)
                    .expect("not exactly one non-1-ZST field in a `DispatchFromDyn` type");
                arg = arg.value_field(fx, idx);
            }
        }

        if let ty::Ref(_, ty, _) = arg.layout().ty.kind() {
            if ty.is_dyn_star() {
                let inner_layout = fx.layout_of(arg.layout().ty.builtin_deref(true).unwrap());
                let dyn_star = CPlace::for_ptr(Pointer::new(arg.load_scalar(fx)), inner_layout);
                let ptr = dyn_star.place_field(fx, FieldIdx::ZERO).to_ptr();
                let vtable = dyn_star.place_field(fx, FieldIdx::ONE).to_cvalue(fx).load_scalar(fx);
                break 'block (ptr, vtable);
            }
        }

        if let BackendRepr::ScalarPair(_, _) = arg.layout().backend_repr {
            let (ptr, vtable) = arg.load_scalar_pair(fx);
            (Pointer::new(ptr), vtable)
        } else {
            let (ptr, vtable) = arg.try_to_ptr().unwrap();
            (ptr, vtable.unwrap())
        }
    };

    let usize_size = fx.layout_of(fx.tcx.types.usize).size.bytes();
    let func_ref = fx.bcx.ins().load(
        fx.pointer_type,
        vtable_memflags(),
        vtable,
        (idx * usize_size as usize) as i32,
    );
    (ptr, func_ref)
}

pub(crate) fn get_vtable<'tcx>(
    fx: &mut FunctionCx<'_, '_, 'tcx>,
    ty: Ty<'tcx>,
    trait_ref: Option<ty::ExistentialTraitRef<'tcx>>,
) -> Value {
    let data_id = data_id_for_vtable(fx.tcx, &mut fx.constants_cx, fx.module, ty, trait_ref);
    let local_data_id = fx.module.declare_data_in_func(data_id, fx.bcx.func);
    if fx.clif_comments.enabled() {
        fx.add_comment(local_data_id, "vtable");
    }
    fx.bcx.ins().global_value(fx.pointer_type, local_data_id)
}
