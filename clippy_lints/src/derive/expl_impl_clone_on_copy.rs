use clippy_utils::diagnostics::span_lint_hir_and_then;
use clippy_utils::ty::{implements_trait, is_copy};
use rustc_hir::{self as hir, HirId, Item};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, GenericArgKind, Ty};

use super::EXPL_IMPL_CLONE_ON_COPY;

/// Implementation of the `EXPL_IMPL_CLONE_ON_COPY` lint.
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    item: &Item<'_>,
    trait_ref: &hir::TraitRef<'_>,
    ty: Ty<'tcx>,
    adt_hir_id: HirId,
) {
    let clone_id = match cx.tcx.lang_items().clone_trait() {
        Some(id) if trait_ref.trait_def_id() == Some(id) => id,
        _ => return,
    };
    let Some(copy_id) = cx.tcx.lang_items().copy_trait() else {
        return;
    };
    let (ty_adt, ty_subs) = match *ty.kind() {
        // Unions can't derive clone.
        ty::Adt(adt, subs) if !adt.is_union() => (adt, subs),
        _ => return,
    };
    // If the current self type doesn't implement Copy (due to generic constraints), search to see if
    // there's a Copy impl for any instance of the adt.
    if !is_copy(cx, ty) {
        if ty_subs.non_erasable_generics().next().is_some() {
            let has_copy_impl = cx.tcx.local_trait_impls(copy_id).iter().any(|&id| {
                matches!(cx.tcx.type_of(id).instantiate_identity().kind(), ty::Adt(adt, _)
                                        if ty_adt.did() == adt.did())
            });
            if !has_copy_impl {
                return;
            }
        } else {
            return;
        }
    }
    // Derive constrains all generic types to requiring Clone. Check if any type is not constrained for
    // this impl.
    if ty_subs.types().any(|ty| !implements_trait(cx, ty, clone_id, &[])) {
        return;
    }
    // `#[repr(packed)]` structs with type/const parameters can't derive `Clone`.
    // https://github.com/rust-lang/rust-clippy/issues/10188
    if ty_adt.repr().packed()
        && ty_subs
            .iter()
            .any(|arg| matches!(arg.kind(), GenericArgKind::Type(_) | GenericArgKind::Const(_)))
    {
        return;
    }
    // The presence of `unsafe` fields prevents deriving `Clone` automatically
    if ty_adt.all_fields().any(|f| f.safety.is_unsafe()) {
        return;
    }

    span_lint_hir_and_then(
        cx,
        EXPL_IMPL_CLONE_ON_COPY,
        adt_hir_id,
        item.span,
        "you are implementing `Clone` explicitly on a `Copy` type",
        |diag| {
            diag.span_help(item.span, "consider deriving `Clone` or removing `Copy`");
        },
    );
}
