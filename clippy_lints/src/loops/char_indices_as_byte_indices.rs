use std::ops::ControlFlow;

use clippy_utils::diagnostics::span_lint_hir_and_then;
use clippy_utils::ty::is_type_lang_item;
use clippy_utils::visitors::for_each_expr;
use clippy_utils::{eq_expr_value, higher, path_to_local_id, sym};
use rustc_errors::{Applicability, MultiSpan};
use rustc_hir::{Expr, ExprKind, LangItem, Node, Pat, PatKind};
use rustc_lint::LateContext;
use rustc_middle::ty::Ty;
use rustc_span::{Span, Symbol};

use super::CHAR_INDICES_AS_BYTE_INDICES;

// The list of `str` methods we want to lint that have a `usize` argument representing a byte index.
// Note: `String` also has methods that work with byte indices,
// but they all take `&mut self` and aren't worth considering since the user couldn't have called
// them while the chars iterator is live anyway.
const BYTE_INDEX_METHODS: &[Symbol] = &[
    sym::ceil_char_boundary,
    sym::floor_char_boundary,
    sym::get,
    sym::get_mut,
    sym::get_unchecked,
    sym::get_unchecked_mut,
    sym::index,
    sym::index_mut,
    sym::is_char_boundary,
    sym::slice_mut_unchecked,
    sym::slice_unchecked,
    sym::split_at,
    sym::split_at_checked,
    sym::split_at_mut,
    sym::split_at_mut_checked,
];

const CONTINUE: ControlFlow<!, ()> = ControlFlow::Continue(());

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, pat: &Pat<'_>, iterable: &Expr<'_>, body: &'tcx Expr<'tcx>) {
    if let ExprKind::MethodCall(_, enumerate_recv, _, enumerate_span) = iterable.kind
        && let Some(method_id) = cx.typeck_results().type_dependent_def_id(iterable.hir_id)
        && cx.tcx.is_diagnostic_item(sym::enumerate_method, method_id)
        && let ExprKind::MethodCall(_, chars_recv, _, chars_span) = enumerate_recv.kind
        && let Some(method_id) = cx.typeck_results().type_dependent_def_id(enumerate_recv.hir_id)
        && cx.tcx.is_diagnostic_item(sym::str_chars, method_id)
    {
        if let PatKind::Tuple([pat, _], _) = pat.kind
            && let PatKind::Binding(_, binding_id, ..) = pat.kind
        {
            // Destructured iterator element `(idx, _)`, look for uses of the binding
            for_each_expr(cx, body, |expr| {
                if path_to_local_id(expr, binding_id) {
                    check_index_usage(cx, expr, pat, enumerate_span, chars_span, chars_recv);
                }
                CONTINUE
            });
        } else if let PatKind::Binding(_, binding_id, ..) = pat.kind {
            // Bound as a tuple, look for `tup.0`
            for_each_expr(cx, body, |expr| {
                if let ExprKind::Field(e, field) = expr.kind
                    && path_to_local_id(e, binding_id)
                    && field.name == sym::integer(0)
                {
                    check_index_usage(cx, expr, pat, enumerate_span, chars_span, chars_recv);
                }
                CONTINUE
            });
        }
    }
}

fn check_index_usage<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    pat: &Pat<'_>,
    enumerate_span: Span,
    chars_span: Span,
    chars_recv: &Expr<'_>,
) {
    let Some(parent_expr) = index_consumed_at(cx, expr) else {
        return;
    };

    let is_string_like = |ty: Ty<'_>| ty.is_str() || is_type_lang_item(cx, ty, LangItem::String);
    let message = match parent_expr.kind {
        ExprKind::MethodCall(segment, recv, ..)
            // We currently only lint `str` methods (which `String` can deref to), so a `.is_str()` check is sufficient here
            // (contrary to the `ExprKind::Index` case which needs to handle both with `is_string_like` because `String` implements
            // `Index` directly and no deref to `str` would happen in that case).
            if cx.typeck_results().expr_ty_adjusted(recv).peel_refs().is_str()
                && BYTE_INDEX_METHODS.contains(&segment.ident.name)
                && eq_expr_value(cx, chars_recv, recv) =>
        {
            "passing a character position to a method that expects a byte index"
        },
        ExprKind::Index(target, ..)
            if is_string_like(cx.typeck_results().expr_ty_adjusted(target).peel_refs())
                && eq_expr_value(cx, chars_recv, target) =>
        {
            "indexing into a string with a character position where a byte index is expected"
        },
        _ => return,
    };

    span_lint_hir_and_then(
        cx,
        CHAR_INDICES_AS_BYTE_INDICES,
        expr.hir_id,
        expr.span,
        message,
        |diag| {
            diag.note("a character can take up more than one byte, so they are not interchangeable")
                .span_note(
                    MultiSpan::from_spans(vec![pat.span, enumerate_span]),
                    "position comes from the enumerate iterator",
                )
                .span_suggestion_verbose(
                    chars_span.to(enumerate_span),
                    "consider using `.char_indices()` instead",
                    "char_indices()",
                    Applicability::MaybeIncorrect,
                );
        },
    );
}

/// Returns the expression which ultimately consumes the index.
/// This is usually the parent expression, i.e. `.split_at(idx)` for `idx`,
/// but for `.get(..idx)` we want to consider the method call the consuming expression,
/// which requires skipping past the range expression.
fn index_consumed_at<'tcx>(cx: &LateContext<'tcx>, expr: &'tcx Expr<'tcx>) -> Option<&'tcx Expr<'tcx>> {
    for (_, node) in cx.tcx.hir_parent_iter(expr.hir_id) {
        match node {
            Node::Expr(expr) if higher::Range::hir(expr).is_some() => {},
            Node::ExprField(_) => {},
            Node::Expr(expr) => return Some(expr),
            _ => break,
        }
    }
    None
}
