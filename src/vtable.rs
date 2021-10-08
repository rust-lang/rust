//! Codegen vtables and vtable accesses.
//!
//! See `rustc_codegen_ssa/src/meth.rs` for reference.

use crate::constant::data_id_for_alloc_id;
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
    arg: CValue<'tcx>,
    idx: usize,
) -> (Value, Value) {
    let (ptr, vtable) = if let Abi::ScalarPair(_, _) = arg.layout().abi {
        arg.load_scalar_pair(fx)
    } else {
        let (ptr, vtable) = arg.try_to_ptr().unwrap();
        (ptr.get_addr(fx), vtable.unwrap())
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
    trait_ref: Option<ty::PolyExistentialTraitRef<'tcx>>,
) -> Value {
    let alloc_id = fx.tcx.vtable_allocation((ty, trait_ref));
    let data_id =
        data_id_for_alloc_id(&mut fx.constants_cx, &mut *fx.module, alloc_id, Mutability::Not);
    let local_data_id = fx.module.declare_data_in_func(data_id, &mut fx.bcx.func);
    if fx.clif_comments.enabled() {
        fx.add_comment(local_data_id, format!("vtable: {:?}", alloc_id));
    }
    fx.bcx.ins().global_value(fx.pointer_type, local_data_id)
}
