use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_with_applicability;
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
        .is_some_and(|id| implements_trait(cx, cx.typeck_results.expr_ty(recv), id, &[]))
    {
        return;
    }
    span_lint_and_then(
        cx,
        FILTER_NEXT,
        expr.span,
        format!("called `filter(..).{next_method}()` on an `{required_trait}`"),
        |diag| {
            let mut app = Applicability::MachineApplicable;
            let filter_snippet = snippet_with_applicability(cx, filter_arg.span, "..", &mut app);
            let iter_snippet = snippet_with_applicability(cx, recv.span, "..", &mut app);

            let pat = if let Some(id) = path_to_local_with_projections(recv)
                && let Node::Pat(pat) = cx.tcx.hir_node(id)
                && let PatKind::Binding(BindingMode(_, Mutability::Not), _, ident, _) = pat.kind
            {
                app = Applicability::Unspecified;
                Some((pat.span, ident))
            } else {
                None
            };

            diag.span_suggestion_verbose(
                expr.span,
                format!("use `.{find_method}(..)` instead"),
                format!("{iter_snippet}.{find_method}({filter_snippet})"),
                app,
            );

            if let Some((pat_span, ident)) = pat {
                diag.span_help(
                    pat_span,
                    format!("you will also need to make `{ident}` mutable, because `{find_method}` takes `&mut self`"),
                );
            }
        },
    );
}
