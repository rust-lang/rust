use super::WRONG_TRANSMUTE;
use clippy_utils::diagnostics::span_lint;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};

/// Checks for `wrong_transmute` lint.
/// Returns `true` if it's triggered, otherwise returns `false`.
pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, e: &'tcx Expr<'_>, from_ty: Ty<'tcx>, to_ty: Ty<'tcx>) -> bool {
    match (&from_ty.kind(), &to_ty.kind()) {
        (ty::Float(_) | ty::Char, ty::Ref(..) | ty::RawPtr(_, _)) => {
            span_lint(
                cx,
                WRONG_TRANSMUTE,
                e.span,
                format!("transmute from a `{from_ty}` to a pointer"),
            );
            true
        },
        _ => false,
    }
}
