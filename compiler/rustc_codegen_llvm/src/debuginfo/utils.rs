// Utility Functions.

use super::namespace::item_namespace;
use super::CrateDebugContext;

use rustc_hir::def_id::DefId;
use rustc_middle::ty::layout::LayoutOf;
use rustc_middle::ty::{self, DefIdTree, Ty};
use rustc_target::abi::Variants;

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

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum FatPtrKind {
    Slice,
    Dyn,
}

/// Determines if `pointee_ty` is slice-like or trait-object-like, i.e.
/// if the second field of the fat pointer is a length or a vtable-pointer.
/// If `pointee_ty` does not require a fat pointer (because it is Sized) then
/// the function returns `None`.
pub(crate) fn fat_pointer_kind<'ll, 'tcx>(
    cx: &CodegenCx<'ll, 'tcx>,
    pointee_ty: Ty<'tcx>,
) -> Option<FatPtrKind> {
    let layout = cx.layout_of(pointee_ty);

    if !layout.is_unsized() {
        return None;
    }

    match *pointee_ty.kind() {
        ty::Str | ty::Slice(_) => Some(FatPtrKind::Slice),
        ty::Dynamic(..) => Some(FatPtrKind::Dyn),
        ty::Adt(..) | ty::Tuple(..) if matches!(layout.variants, Variants::Single { .. }) => {
            let last_field_index = layout.fields.count() - 1;
            debug_assert!(
                (0..last_field_index)
                    .all(|field_index| { !layout.field(cx, field_index).is_unsized() })
            );

            let unsized_field = layout.field(cx, last_field_index);
            debug_assert!(unsized_field.is_unsized());
            fat_pointer_kind(cx, unsized_field.ty)
        }
        ty::Foreign(_) => {
            // Assert that pointers to foreign types really are thin:
            debug_assert_eq!(
                cx.size_of(cx.tcx.mk_imm_ptr(pointee_ty)),
                cx.size_of(cx.tcx.mk_imm_ptr(cx.tcx.types.u8))
            );
            None
        }
        _ => {
            // For all other pointee types we should already have returned None
            // at the beginning of the function.
            panic!("fat_pointer_kind() - Encountered unexpected `pointee_ty`: {:?}", pointee_ty)
        }
    }
}
