// Utility Functions.

use rustc_hir::def_id::DefId;
use rustc_middle::ty::layout::{HasTypingEnv, LayoutOf};
use rustc_middle::ty::{self, Ty};
use tracing::trace;

use super::CodegenUnitDebugContext;
use super::namespace::item_namespace;
use crate::common::CodegenCx;
use crate::llvm;
use crate::llvm::debuginfo::{DIArray, DIBuilder, DIDescriptor, DIScope};

pub(crate) fn is_node_local_to_unit(cx: &CodegenCx<'_, '_>, def_id: DefId) -> bool {
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
pub(crate) fn create_DIArray<'ll>(
    builder: &DIBuilder<'ll>,
    arr: &[Option<&'ll DIDescriptor>],
) -> &'ll DIArray {
    unsafe { llvm::LLVMDIBuilderGetOrCreateArray(builder, arr.as_ptr(), arr.len()) }
}

#[inline]
pub(crate) fn debug_context<'a, 'll, 'tcx>(
    cx: &'a CodegenCx<'ll, 'tcx>,
) -> &'a CodegenUnitDebugContext<'ll, 'tcx> {
    cx.dbg_cx.as_ref().unwrap()
}

#[inline]
#[allow(non_snake_case)]
pub(crate) fn DIB<'a, 'll>(cx: &'a CodegenCx<'ll, '_>) -> &'a DIBuilder<'ll> {
    cx.dbg_cx.as_ref().unwrap().builder.as_ref()
}

pub(crate) fn get_namespace_for_item<'ll>(cx: &CodegenCx<'ll, '_>, def_id: DefId) -> &'ll DIScope {
    item_namespace(cx, cx.tcx.parent(def_id))
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum WidePtrKind {
    Slice,
    Dyn,
}

/// Determines if `pointee_ty` is slice-like or trait-object-like, i.e.
/// if the second field of the wide pointer is a length or a vtable-pointer.
/// If `pointee_ty` does not require a wide pointer (because it is Sized) then
/// the function returns `None`.
pub(crate) fn wide_pointer_kind<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    pointee_ty: Ty<'tcx>,
) -> Option<WidePtrKind> {
    let pointee_tail_ty = cx.tcx.struct_tail_for_codegen(pointee_ty, cx.typing_env());
    let layout = cx.layout_of(pointee_tail_ty);
    trace!(
        "wide_pointer_kind: {:?} has layout {:?} (is_unsized? {})",
        pointee_tail_ty,
        layout,
        layout.is_unsized()
    );

    if layout.is_sized() {
        return None;
    }

    match *pointee_tail_ty.kind() {
        ty::Str | ty::Slice(_) => Some(WidePtrKind::Slice),
        ty::Dynamic(..) => Some(WidePtrKind::Dyn),
        ty::Foreign(_) => {
            // Assert that pointers to foreign types really are thin:
            assert_eq!(
                cx.size_of(Ty::new_imm_ptr(cx.tcx, pointee_tail_ty)),
                cx.size_of(Ty::new_imm_ptr(cx.tcx, cx.tcx.types.u8))
            );
            None
        }
        _ => {
            // For all other pointee types we should already have returned None
            // at the beginning of the function.
            panic!(
                "wide_pointer_kind() - Encountered unexpected `pointee_tail_ty`: {pointee_tail_ty:?}"
            )
        }
    }
}
