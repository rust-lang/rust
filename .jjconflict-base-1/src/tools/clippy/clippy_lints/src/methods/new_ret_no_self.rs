use clippy_utils::diagnostics::span_lint;
use clippy_utils::return_ty;
use clippy_utils::ty::contains_ty_adt_constructor_opaque;
use rustc_hir::{ImplItem, TraitItem};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};
use rustc_span::sym;

use super::NEW_RET_NO_SELF;

pub(super) fn check_impl_item<'tcx>(
    cx: &LateContext<'tcx>,
    impl_item: &'tcx ImplItem<'_>,
    self_ty: Ty<'tcx>,
    implements_trait: bool,
) {
    // if this impl block implements a trait, lint in trait definition instead
    if !implements_trait
        && impl_item.ident.name == sym::new
        && let ret_ty = return_ty(cx, impl_item.owner_id)
        && ret_ty != self_ty
        && !contains_ty_adt_constructor_opaque(cx, ret_ty, self_ty)
    {
        span_lint(
            cx,
            NEW_RET_NO_SELF,
            impl_item.span,
            "methods called `new` usually return `Self`",
        );
    }
}

pub(super) fn check_trait_item<'tcx>(cx: &LateContext<'tcx>, trait_item: &'tcx TraitItem<'tcx>) {
    if trait_item.ident.name == sym::new
        && let ret_ty = return_ty(cx, trait_item.owner_id)
        && let self_ty = ty::TraitRef::identity(cx.tcx, trait_item.owner_id.to_def_id()).self_ty()
        && !ret_ty.contains(self_ty)
    {
        span_lint(
            cx,
            NEW_RET_NO_SELF,
            trait_item.span,
            "methods called `new` usually return `Self`",
        );
    }
}
