// Utility Functions.

use super::namespace::item_namespace;
use super::CrateDebugContext;

use rustc_hir::def_id::DefId;
use rustc_middle::ty::DefIdTree;

use crate::common::CodegenCx;
use crate::llvm;
use crate::llvm::debuginfo::{DIArray, DIBuilder, DIDescriptor, DIScope};

pub fn is_node_local_to_unit(cx: &CodegenCx<'_, '_>, def_id: DefId) -> bool {
    // The is_local_to_unit flag indicates whether a function is local to the
    // current compilation unit (i.e., if it is *static* in the C-sense). The
    // *reachable* set should provide a good approximation of this, as it
    // contains everything that might leak out of the current crate (by being
    // externally visible or by being inlined into something externally
    // visible). It might better to use the `exported_items` set from
    // `driver::CrateAnalysis` in the future, but (atm) this set is not
    // available in the codegen pass.
    !cx.tcx.is_reachable_non_generic(def_id)
}

#[allow(non_snake_case)]
pub fn create_DIArray<'ll>(
    builder: &DIBuilder<'ll>,
    arr: &[Option<&'ll DIDescriptor>],
) -> &'ll DIArray {
    unsafe { llvm::LLVMRustDIBuilderGetOrCreateArray(builder, arr.as_ptr(), arr.len() as u32) }
}

#[inline]
pub fn debug_context<'a, 'll, 'tcx>(
    cx: &'a CodegenCx<'ll, 'tcx>,
) -> &'a CrateDebugContext<'ll, 'tcx> {
    cx.dbg_cx.as_ref().unwrap()
}

#[inline]
#[allow(non_snake_case)]
pub fn DIB<'a, 'll>(cx: &'a CodegenCx<'ll, '_>) -> &'a DIBuilder<'ll> {
    cx.dbg_cx.as_ref().unwrap().builder
}

pub fn get_namespace_for_item<'ll>(cx: &CodegenCx<'ll, '_>, def_id: DefId) -> &'ll DIScope {
    item_namespace(cx, cx.tcx.parent(def_id).expect("get_namespace_for_item: missing parent?"))
}
