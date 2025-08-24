use super::USELESS_TRANSMUTE;
use clippy_utils::diagnostics::{span_lint, span_lint_and_then};
use clippy_utils::sugg;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty, TypeVisitableExt};

/// Checks for `useless_transmute` lint.
/// Returns `true` if it's triggered, otherwise returns `false`.
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    e: &'tcx Expr<'_>,
    from_ty: Ty<'tcx>,
    to_ty: Ty<'tcx>,
    arg: &'tcx Expr<'_>,
) -> bool {
    match (*from_ty.kind(), *to_ty.kind()) {
        _ if from_ty == to_ty && !from_ty.has_erased_regions() => {
            span_lint(
                cx,
                USELESS_TRANSMUTE,
                e.span,
                format!("transmute from a type (`{from_ty}`) to itself"),
            );
            true
        },
        (ty::Ref(_, rty, rty_mutbl), ty::RawPtr(ptr_ty, ptr_mutbl)) => {
            // No way to give the correct suggestion here. Avoid linting for now.
            if !rty.has_erased_regions() {
                span_lint_and_then(
                    cx,
                    USELESS_TRANSMUTE,
                    e.span,
                    "transmute from a reference to a pointer",
                    |diag| {
                        if let Some(arg) = sugg::Sugg::hir_opt(cx, arg) {
                            let sugg = if ptr_ty == rty && rty_mutbl == ptr_mutbl {
                                arg.as_ty(to_ty)
                            } else {
                                arg.as_ty(Ty::new_ptr(cx.tcx, rty, rty_mutbl)).as_ty(to_ty)
                            };

                            diag.span_suggestion(e.span, "try", sugg, Applicability::Unspecified);
                        }
                    },
                );
            }
            true
        },
        (ty::Int(_) | ty::Uint(_), ty::RawPtr(_, _)) => {
            // Handled by the upstream rustc `integer_to_ptr_transmutes` lint
            true
        },
        _ => false,
    }
}
