use super::TRANSMUTE_PTR_TO_PTR;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::sugg;
use rustc_errors::Applicability;
use rustc_hir::Expr;
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty, TypeVisitableExt};

/// Checks for `transmute_ptr_to_ptr` lint.
/// Returns `true` if it's triggered, otherwise returns `false`.
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    e: &'tcx Expr<'_>,
    from_ty: Ty<'tcx>,
    to_ty: Ty<'tcx>,
    arg: &'tcx Expr<'_>,
    msrv: Msrv,
) -> bool {
    match (from_ty.kind(), to_ty.kind()) {
        (ty::RawPtr(from_pointee_ty, from_mutbl), ty::RawPtr(to_pointee_ty, to_mutbl)) => {
            span_lint_and_then(
                cx,
                TRANSMUTE_PTR_TO_PTR,
                e.span,
                "transmute from a pointer to a pointer",
                |diag| {
                    if let Some(arg) = sugg::Sugg::hir_opt(cx, arg) {
                        if from_mutbl == to_mutbl
                            && to_pointee_ty.is_sized(cx.tcx, cx.typing_env())
                            && msrv.meets(cx, msrvs::POINTER_CAST)
                        {
                            diag.span_suggestion_verbose(
                                e.span,
                                "use `pointer::cast` instead",
                                format!("{}.cast::<{to_pointee_ty}>()", arg.maybe_paren()),
                                Applicability::MaybeIncorrect,
                            );
                        } else if from_pointee_ty == to_pointee_ty
                            && let Some(method) = match (from_mutbl, to_mutbl) {
                                (ty::Mutability::Not, ty::Mutability::Mut) => Some("cast_mut"),
                                (ty::Mutability::Mut, ty::Mutability::Not) => Some("cast_const"),
                                _ => None,
                            }
                            && !from_pointee_ty.has_erased_regions()
                            && msrv.meets(cx, msrvs::POINTER_CAST_CONSTNESS)
                        {
                            diag.span_suggestion_verbose(
                                e.span,
                                format!("use `pointer::{method}` instead"),
                                format!("{}.{method}()", arg.maybe_paren()),
                                Applicability::MaybeIncorrect,
                            );
                        } else {
                            diag.span_suggestion_verbose(
                                e.span,
                                "use an `as` cast instead",
                                arg.as_ty(to_ty),
                                Applicability::MaybeIncorrect,
                            );
                        }
                    }
                },
            );
            true
        },
        _ => false,
    }
}
