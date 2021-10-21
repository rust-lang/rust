use super::TRANSMUTE_NUM_TO_BYTES;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::sugg;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty, UintTy};

/// Checks for `transmute_int_to_float` lint.
/// Returns `true` if it's triggered, otherwise returns `false`.
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    e: &'tcx Expr<'_>,
    from_ty: Ty<'tcx>,
    to_ty: Ty<'tcx>,
    args: &'tcx [Expr<'_>],
    const_context: bool,
) -> bool {
    match (&from_ty.kind(), &to_ty.kind()) {
        (ty::Int(_) | ty::Uint(_) | ty::Float(_), ty::Array(arr_ty, _)) => {
            if !matches!(arr_ty.kind(), ty::Uint(UintTy::U8)) {
                return false;
            }
            if matches!(from_ty.kind(), ty::Float(_)) && const_context {
                // TODO: Remove when const_float_bits_conv is stabilized
                // rust#72447
                return false;
            }

            span_lint_and_then(
                cx,
                TRANSMUTE_NUM_TO_BYTES,
                e.span,
                &format!("transmute from a `{}` to a `{}`", from_ty, to_ty),
                |diag| {
                    let arg = sugg::Sugg::hir(cx, &args[0], "..");
                    diag.span_suggestion(
                        e.span,
                        "consider using `to_ne_bytes()`",
                        format!("{}.to_ne_bytes()", arg),
                        Applicability::Unspecified,
                    );
                },
            );
            true
        },
        _ => false,
    }
}
