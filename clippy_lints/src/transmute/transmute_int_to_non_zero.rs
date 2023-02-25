use super::TRANSMUTE_INT_TO_NON_ZERO;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::sugg;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_middle::{
    query::Key,
    ty::{self, Ty},
};
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
    let (ty::Int(_) | ty::Uint(_), Some(to_ty_id)) = (&from_ty.kind(), to_ty.ty_adt_id()) else {
        return false;
    };
    let Some(to_type_sym) = cx.tcx.get_diagnostic_name(to_ty_id) else {
        return false;
    };

    if !matches!(
        to_type_sym,
        sym::NonZeroU8
            | sym::NonZeroU16
            | sym::NonZeroU32
            | sym::NonZeroU64
            | sym::NonZeroU128
            | sym::NonZeroI8
            | sym::NonZeroI16
            | sym::NonZeroI32
            | sym::NonZeroI64
            | sym::NonZeroI128
    ) {
        return false;
    }

    span_lint_and_then(
        cx,
        TRANSMUTE_INT_TO_NON_ZERO,
        e.span,
        &format!("transmute from a `{from_ty}` to a `{to_type_sym}`"),
        |diag| {
            let arg = sugg::Sugg::hir(cx, arg, "..");
            diag.span_suggestion(
                e.span,
                "consider using",
                format!("{to_type_sym}::{}({arg})", sym::new_unchecked),
                Applicability::Unspecified,
            );
        },
    );
    true
}
