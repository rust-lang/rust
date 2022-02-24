use super::utils::get_type_snippet;
use super::TRANSMUTE_PTR_TO_REF;
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::sugg;
use rustc_errors::Applicability;
use rustc_hir::{Expr, Mutability, QPath};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, Ty};

/// Checks for `transmute_ptr_to_ref` lint.
/// Returns `true` if it's triggered, otherwise returns `false`.
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    e: &'tcx Expr<'_>,
    from_ty: Ty<'tcx>,
    to_ty: Ty<'tcx>,
    arg: &'tcx Expr<'_>,
    qpath: &'tcx QPath<'_>,
) -> bool {
    match (&from_ty.kind(), &to_ty.kind()) {
        (ty::RawPtr(from_ptr_ty), ty::Ref(_, to_ref_ty, mutbl)) => {
            span_lint_and_then(
                cx,
                TRANSMUTE_PTR_TO_REF,
                e.span,
                &format!(
                    "transmute from a pointer type (`{}`) to a reference type (`{}`)",
                    from_ty, to_ty
                ),
                |diag| {
                    let arg = sugg::Sugg::hir(cx, arg, "..");
                    let (deref, cast) = if *mutbl == Mutability::Mut {
                        ("&mut *", "*mut")
                    } else {
                        ("&*", "*const")
                    };

                    let arg = if from_ptr_ty.ty == *to_ref_ty {
                        arg
                    } else {
                        arg.as_ty(&format!("{} {}", cast, get_type_snippet(cx, qpath, *to_ref_ty)))
                    };

                    diag.span_suggestion(
                        e.span,
                        "try",
                        sugg::make_unop(deref, arg).to_string(),
                        Applicability::Unspecified,
                    );
                },
            );
            true
        },
        _ => false,
    }
}
