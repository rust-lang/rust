use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet;
use clippy_utils::ty::implements_trait;
use clippy_utils::{path_to_local_with_projections, sym};
use rustc_ast::{BindingMode, Mutability};
use rustc_errors::Applicability;
use rustc_hir::{Expr, Node, PatKind};
use rustc_lint::LateContext;

use super::FILTER_NEXT;

#[derive(Clone, Copy)]
pub(super) enum Direction {
    Forward,
    Backward,
}

/// lint use of `filter().next()` for `Iterator` and `filter().next_back()` for
/// `DoubleEndedIterator`
pub(super) fn check(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    recv: &Expr<'_>,
    filter_arg: &Expr<'_>,
    direction: Direction,
) {
    let (required_trait, next_method, find_method) = match direction {
        Direction::Forward => (sym::Iterator, "next", "find"),
        Direction::Backward => (sym::DoubleEndedIterator, "next_back", "rfind"),
    };
    if !cx
        .tcx
        .get_diagnostic_item(required_trait)
        .is_some_and(|id| implements_trait(cx, cx.typeck_results().expr_ty(recv), id, &[]))
    {
        return;
    }
    span_lint_and_then(
        cx,
        FILTER_NEXT,
        expr.span,
        format!(
            "called `filter(..).{next_method}()` on an `{required_trait}`. \
            This is more succinctly expressed by calling `.{find_method}(..)` instead",
        ),
        |diag| {
            // add note if not multi-line
            let filter_snippet = snippet(cx, filter_arg.span, "..");
            if filter_snippet.lines().count() <= 1 {
                let iter_snippet = snippet(cx, recv.span, "..");
                let (applicability, pat) = if let Some(id) = path_to_local_with_projections(recv)
                    && let Node::Pat(pat) = cx.tcx.hir_node(id)
                    && let PatKind::Binding(BindingMode(_, Mutability::Not), _, ident, _) = pat.kind
                {
                    (Applicability::Unspecified, Some((pat.span, ident)))
                } else {
                    (Applicability::MachineApplicable, None)
                };

                diag.span_suggestion(
                    expr.span,
                    "try",
                    format!("{iter_snippet}.{find_method}({filter_snippet})"),
                    applicability,
                );

                if let Some((pat_span, ident)) = pat {
                    diag.span_help(
                        pat_span,
                        format!(
                            "you will also need to make `{ident}` mutable, because `{find_method}` takes `&mut self`"
                        ),
                    );
                }
            }
        },
    );
}
