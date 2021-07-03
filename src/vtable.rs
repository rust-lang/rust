//! Codegen vtables and vtable accesses.
//!
//! See `rustc_codegen_ssa/src/meth.rs` for reference.
// FIXME dedup this logic between miri, cg_llvm and cg_clif

use crate::prelude::*;
use super::constant::pointer_for_allocation;

fn vtable_memflags() -> MemFlags {
    let mut flags = MemFlags::trusted(); // A vtable access is always aligned and will never trap.
    flags.set_readonly(); // A vtable is always read-only.
    flags
}

pub(crate) fn drop_fn_of_obj(fx: &mut FunctionCx<'_, '_, '_>, vtable: Value) -> Value {
    let usize_size = fx.layout_of(fx.tcx.types.usize).size.bytes() as usize;
    fx.bcx.ins().load(
        pointer_ty(fx.tcx),
        vtable_memflags(),
        vtable,
        (ty::COMMON_VTABLE_ENTRIES_DROPINPLACE * usize_size) as i32,
    )
}

pub(crate) fn size_of_obj(fx: &mut FunctionCx<'_, '_, '_>, vtable: Value) -> Value {
    let usize_size = fx.layout_of(fx.tcx.types.usize).size.bytes() as usize;
    fx.bcx.ins().load(
        pointer_ty(fx.tcx),
        vtable_memflags(),
        vtable,
        (ty::COMMON_VTABLE_ENTRIES_SIZE * usize_size) as i32,
    )
}

pub(crate) fn min_align_of_obj(fx: &mut FunctionCx<'_, '_, '_>, vtable: Value) -> Value {
    let usize_size = fx.layout_of(fx.tcx.types.usize).size.bytes() as usize;
    fx.bcx.ins().load(
        pointer_ty(fx.tcx),
        vtable_memflags(),
        vtable,
        (ty::COMMON_VTABLE_ENTRIES_SIZE * usize_size) as i32,
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
        pointer_ty(fx.tcx),
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
    let vtable_ptr = if let Some(vtable_ptr) = fx.vtables.get(&(ty, trait_ref)) {
        *vtable_ptr
    } else {
        let vtable_alloc_id = fx.tcx.vtable_allocation(ty, trait_ref);
        let vtable_allocation = fx.tcx.global_alloc(vtable_alloc_id).unwrap_memory();
        let vtable_ptr = pointer_for_allocation(fx, vtable_allocation);

        fx.vtables.insert((ty, trait_ref), vtable_ptr);
        vtable_ptr
    };

    vtable_ptr.get_addr(fx)
}
