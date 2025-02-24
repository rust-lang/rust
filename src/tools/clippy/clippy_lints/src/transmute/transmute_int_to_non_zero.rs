use super::TRANSMUTE_INT_TO_NON_ZERO;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::sugg;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};
use rustc_span::symbol::sym;

/// Checks for `transmute_int_to_non_zero` lint.
/// Returns `true` if it's triggered, otherwise returns `false`.
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    e: &'tcx Expr<'_>,
    from_ty: Ty<'tcx>,
    to_ty: Ty<'tcx>,
    arg: &'tcx Expr<'_>,
) -> bool {
    let tcx = cx.tcx;

    let (ty::Int(_) | ty::Uint(_), ty::Adt(adt, substs)) = (&from_ty.kind(), to_ty.kind()) else {
        return false;
    };

    if !tcx.is_diagnostic_item(sym::NonZero, adt.did()) {
        return false;
    }

    let int_ty = substs.type_at(0);
    if from_ty != int_ty {
        return false;
    }

    span_lint_and_then(
        cx,
        TRANSMUTE_INT_TO_NON_ZERO,
        e.span,
        format!("transmute from a `{from_ty}` to a `{}<{int_ty}>`", sym::NonZero),
        |diag| {
            let arg = sugg::Sugg::hir(cx, arg, "..");
            diag.span_suggestion(
                e.span,
                "consider using",
                format!("{}::{}({arg})", sym::NonZero, sym::new_unchecked),
                Applicability::Unspecified,
            );
        },
    );
    true
}
