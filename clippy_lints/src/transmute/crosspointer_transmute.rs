use super::CROSSPOINTER_TRANSMUTE;
use crate::utils::span_lint;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};

/// Checks for `crosspointer_transmute` lint.
/// Returns `true` if it's triggered, otherwise returns `false`.
pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, e: &'tcx Expr<'_>, from_ty: Ty<'tcx>, to_ty: Ty<'tcx>) -> bool {
    match (&from_ty.kind(), &to_ty.kind()) {
        (ty::RawPtr(from_ptr), _) if from_ptr.ty == to_ty => {
            span_lint(
                cx,
                CROSSPOINTER_TRANSMUTE,
                e.span,
                &format!(
                    "transmute from a type (`{}`) to the type that it points to (`{}`)",
                    from_ty, to_ty
                ),
            );
            true
        },
        (_, ty::RawPtr(to_ptr)) if to_ptr.ty == from_ty => {
            span_lint(
                cx,
                CROSSPOINTER_TRANSMUTE,
                e.span,
                &format!(
                    "transmute from a type (`{}`) to a pointer to that type (`{}`)",
                    from_ty, to_ty
                ),
            );
            true
        },
        _ => false,
    }
}
