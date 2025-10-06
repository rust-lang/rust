use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::{SpanRangeExt as _, snippet_with_applicability};
use clippy_utils::{SpanlessEq, get_parent_expr, higher, is_integer_const, is_trait_method, sym};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind, Node, Pat, PatKind, QPath};
use rustc_lint::LateContext;

use super::RANGE_ZIP_WITH_LEN;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>, recv: &'tcx Expr<'_>, zip_arg: &'tcx Expr<'_>) {
    if is_trait_method(cx, expr, sym::Iterator)
        // range expression in `.zip()` call: `0..x.len()`
        && let Some(higher::Range { start: Some(start), end: Some(end), .. }) = higher::Range::hir(zip_arg)
        && is_integer_const(cx, start, 0)
        // `.len()` call
        && let ExprKind::MethodCall(len_path, len_recv, [], _) = end.kind
        && len_path.ident.name == sym::len
        // `.iter()` and `.len()` called on same `Path`
        && let ExprKind::Path(QPath::Resolved(_, iter_path)) = recv.kind
        && let ExprKind::Path(QPath::Resolved(_, len_path)) = len_recv.kind
        && SpanlessEq::new(cx).eq_path_segments(iter_path.segments, len_path.segments)
    {
        span_lint_and_then(
            cx,
            RANGE_ZIP_WITH_LEN,
            expr.span,
            "using `.zip()` with a range and `.len()`",
            |diag| {
                // If the iterator content is consumed by a pattern with exactly two elements, swap
                // the order of those elements. Otherwise, the suggestion will be marked as
                // `Applicability::MaybeIncorrect` (because it will be), and a note will be added
                // to the diagnostic to underline the swapping of the index and the content.
                let pat = methods_pattern(cx, expr).or_else(|| for_loop_pattern(cx, expr));
                let invert_bindings = if let Some(pat) = pat
                    && pat.span.eq_ctxt(expr.span)
                    && let PatKind::Tuple([first, second], _) = pat.kind
                {
                    Some((first.span, second.span))
                } else {
                    None
                };
                let mut app = Applicability::MachineApplicable;
                let mut suggestions = vec![(
                    expr.span,
                    format!(
                        "{}.iter().enumerate()",
                        snippet_with_applicability(cx, recv.span, "_", &mut app)
                    ),
                )];
                if let Some((left, right)) = invert_bindings
                    && let Some(snip_left) = left.get_source_text(cx)
                    && let Some(snip_right) = right.get_source_text(cx)
                {
                    suggestions.extend([(left, snip_right.to_string()), (right, snip_left.to_string())]);
                } else {
                    app = Applicability::MaybeIncorrect;
                }
                diag.multipart_suggestion("use", suggestions, app);
                if app != Applicability::MachineApplicable {
                    diag.note("the order of the element and the index will be swapped");
                }
            },
        );
    }
}

/// If `expr` is the argument of a `for` loop, return the loop pattern.
fn for_loop_pattern<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) -> Option<&'tcx Pat<'tcx>> {
    cx.tcx.hir_parent_iter(expr.hir_id).find_map(|(_, node)| {
        if let Node::Expr(ancestor_expr) = node
            && let Some(for_loop) = higher::ForLoop::hir(ancestor_expr)
            && for_loop.arg.hir_id == expr.hir_id
        {
            Some(for_loop.pat)
        } else {
            None
        }
    })
}

/// If `expr` is the receiver of an `Iterator` method which consumes the iterator elements and feed
/// them to a closure, return the pattern of the closure.
fn methods_pattern<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) -> Option<&'tcx Pat<'tcx>> {
    if let Some(parent_expr) = get_parent_expr(cx, expr)
        && is_trait_method(cx, expr, sym::Iterator)
        && let ExprKind::MethodCall(method, recv, [arg], _) = parent_expr.kind
        && recv.hir_id == expr.hir_id
        && matches!(
            method.ident.name,
            sym::all
                | sym::any
                | sym::filter_map
                | sym::find_map
                | sym::flat_map
                | sym::for_each
                | sym::is_partitioned
                | sym::is_sorted_by_key
                | sym::map
                | sym::map_while
                | sym::position
                | sym::rposition
                | sym::try_for_each
        )
        && let ExprKind::Closure(closure) = arg.kind
        && let body = cx.tcx.hir_body(closure.body)
        && let [param] = body.params
    {
        Some(param.pat)
    } else {
        None
    }
}
