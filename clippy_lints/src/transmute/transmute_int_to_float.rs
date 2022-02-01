use super::TRANSMUTE_INT_TO_FLOAT;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::sugg;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};

/// Checks for `transmute_int_to_float` lint.
/// Returns `true` if it's triggered, otherwise returns `false`.
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    e: &'tcx Expr<'_>,
    from_ty: Ty<'tcx>,
    to_ty: Ty<'tcx>,
    arg: &'tcx Expr<'_>,
    const_context: bool,
) -> bool {
    match (&from_ty.kind(), &to_ty.kind()) {
        (ty::Int(_) | ty::Uint(_), ty::Float(_)) if !const_context => {
            span_lint_and_then(
                cx,
                TRANSMUTE_INT_TO_FLOAT,
                e.span,
                &format!("transmute from a `{}` to a `{}`", from_ty, to_ty),
                |diag| {
                    let arg = sugg::Sugg::hir(cx, arg, "..");
                    let arg = if let ty::Int(int_ty) = from_ty.kind() {
                        arg.as_ty(format!(
                            "u{}",
                            int_ty.bit_width().map_or_else(|| "size".to_string(), |v| v.to_string())
                        ))
                    } else {
                        arg
                    };
                    diag.span_suggestion(
                        e.span,
                        "consider using",
                        format!("{}::from_bits({})", to_ty, arg),
                        Applicability::Unspecified,
                    );
                },
            );
            true
        },
        _ => false,
    }
}
