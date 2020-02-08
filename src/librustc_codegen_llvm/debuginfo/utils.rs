// Utility Functions.

use super::namespace::item_namespace;
use super::CrateDebugContext;

use rustc::ty::{DefIdTree, TyCtxt};
use rustc_hir::def_id::DefId;

use crate::common::CodegenCx;
use crate::llvm;
use crate::llvm::debuginfo::{DIArray, DIBuilder, DIDescriptor, DIScope};

use rustc_span::Span;

pub fn is_node_local_to_unit(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    // The is_local_to_unit flag indicates whether a function is local to the
    // current compilation unit (i.e., if it is *static* in the C-sense). The
    // *reachable* set should provide a good approximation of this, as it
    // contains everything that might leak out of the current crate (by being
    // externally visible or by being inlined into something externally
    // visible). It might better to use the `exported_items` set from
    // `driver::CrateAnalysis` in the future, but (atm) this set is not
    // available in the codegen pass.
    !tcx.is_reachable_non_generic(def_id)
}

#[allow(non_snake_case)]
pub fn create_DIArray(builder: &DIBuilder<'ll>, arr: &[Option<&'ll DIDescriptor>]) -> &'ll DIArray {
    return unsafe {
        llvm::LLVMRustDIBuilderGetOrCreateArray(builder, arr.as_ptr(), arr.len() as u32)
    };
}

/// Returns rustc_span::Loc corresponding to the beginning of the span
pub fn span_start(tcx: TyCtxt<'_>, span: Span) -> rustc_span::Loc {
    tcx.sess.source_map().lookup_char_pos(span.lo())
}

#[inline]
pub fn debug_context(cx: &'a CodegenCx<'ll, 'tcx>) -> &'a CrateDebugContext<'ll, 'tcx> {
    cx.dbg_cx.as_ref().unwrap()
}

#[inline]
#[allow(non_snake_case)]
pub fn DIB(cx: &'a CodegenCx<'ll, '_>) -> &'a DIBuilder<'ll> {
    cx.dbg_cx.as_ref().unwrap().builder
}

pub fn get_namespace_for_item(dbg_cx: &CrateDebugContext<'ll, '_>, def_id: DefId) -> &'ll DIScope {
    item_namespace(dbg_cx, dbg_cx.tcx.parent(def_id).expect("get_namespace_for_item: missing parent?"))
}
