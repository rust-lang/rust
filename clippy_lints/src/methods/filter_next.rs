use clippy_utils::diagnostics::{span_lint, span_lint_and_then};
use clippy_utils::path_to_local_with_projections;
use clippy_utils::source::snippet;
use clippy_utils::ty::implements_trait;
use rustc_ast::{BindingMode, Mutability};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::sym;

use super::FILTER_NEXT;

/// lint use of `filter().next()` for `Iterators`
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'_>,
    recv: &'tcx hir::Expr<'_>,
    filter_arg: &'tcx hir::Expr<'_>,
) {
    // lint if caller of `.filter().next()` is an Iterator
    let recv_impls_iterator = cx
        .tcx
        .get_diagnostic_item(sym::Iterator)
        .is_some_and(|id| implements_trait(cx, cx.typeck_results().expr_ty(recv), id, &[]));
    if recv_impls_iterator {
        let msg = "called `filter(..).next()` on an `Iterator`. This is more succinctly expressed by calling \
                   `.find(..)` instead";
        let filter_snippet = snippet(cx, filter_arg.span, "..");
        if filter_snippet.lines().count() <= 1 {
            let iter_snippet = snippet(cx, recv.span, "..");
            // add note if not multi-line
            span_lint_and_then(cx, FILTER_NEXT, expr.span, msg, |diag| {
                let (applicability, pat) = if let Some(id) = path_to_local_with_projections(recv)
                    && let hir::Node::Pat(pat) = cx.tcx.hir_node(id)
                    && let hir::PatKind::Binding(BindingMode(_, Mutability::Not), _, ident, _) = pat.kind
                {
                    (Applicability::Unspecified, Some((pat.span, ident)))
                } else {
                    (Applicability::MachineApplicable, None)
                };

                diag.span_suggestion(
                    expr.span,
                    "try",
                    format!("{iter_snippet}.find({filter_snippet})"),
                    applicability,
                );

                if let Some((pat_span, ident)) = pat {
                    diag.span_help(
                        pat_span,
                        format!("you will also need to make `{ident}` mutable, because `find` takes `&mut self`"),
                    );
                }
            });
        } else {
            span_lint(cx, FILTER_NEXT, expr.span, msg);
        }
    }
}
