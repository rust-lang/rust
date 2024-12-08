use super::TRANSMUTE_NUM_TO_BYTES;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{self, Msrv};
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
    arg: &'tcx Expr<'_>,
    const_context: bool,
    msrv: &Msrv,
) -> bool {
    match (&from_ty.kind(), &to_ty.kind()) {
        (ty::Int(_) | ty::Uint(_) | ty::Float(_), ty::Array(arr_ty, _)) => {
            if !matches!(arr_ty.kind(), ty::Uint(UintTy::U8)) {
                return false;
            }
            if matches!(from_ty.kind(), ty::Float(_)) && const_context && !msrv.meets(msrvs::CONST_FLOAT_BITS_CONV) {
                return false;
            }

            span_lint_and_then(
                cx,
                TRANSMUTE_NUM_TO_BYTES,
                e.span,
                format!("transmute from a `{from_ty}` to a `{to_ty}`"),
                |diag| {
                    let arg = sugg::Sugg::hir(cx, arg, "..");
                    diag.span_suggestion(
                        e.span,
                        "consider using `to_ne_bytes()`",
                        format!("{arg}.to_ne_bytes()"),
                        Applicability::Unspecified,
                    );
                },
            );
            true
        },
        _ => false,
    }
}
