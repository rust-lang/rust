use rustc_errors::Applicability;
use rustc_hir::{Closure, Expr, ExprKind, HirId, StmtKind, UnOp};
use rustc_lint::LateContext;
use rustc_middle::ty;
use rustc_span::Span;

use super::NEEDLESS_CHARACTER_ITERATION;
use super::utils::get_last_chain_binding_hir_id;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::SpanRangeExt;
use clippy_utils::{is_path_diagnostic_item, path_to_local_id, peel_blocks, sym};

fn peels_expr_ref<'a, 'tcx>(mut expr: &'a Expr<'tcx>) -> &'a Expr<'tcx> {
    while let ExprKind::AddrOf(_, _, e) = expr.kind {
        expr = e;
    }
    expr
}

fn handle_expr(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    first_param: HirId,
    span: Span,
    before_chars: Span,
    revert: bool,
    is_all: bool,
) {
    match expr.kind {
        ExprKind::MethodCall(method, receiver, [], _) => {
            // If we have `!is_ascii`, then only `.any()` should warn. And if the condition is
            // `is_ascii`, then only `.all()` should warn.
            if revert != is_all
                && method.ident.name == sym::is_ascii
                && path_to_local_id(receiver, first_param)
                && let char_arg_ty = cx.typeck_results().expr_ty_adjusted(receiver).peel_refs()
                && *char_arg_ty.kind() == ty::Char
                && let Some(snippet) = before_chars.get_source_text(cx)
            {
                span_lint_and_sugg(
                    cx,
                    NEEDLESS_CHARACTER_ITERATION,
                    span,
                    "checking if a string is ascii using iterators",
                    "try",
                    format!("{}{snippet}.is_ascii()", if revert { "!" } else { "" }),
                    Applicability::MachineApplicable,
                );
            }
        },
        ExprKind::Block(block, _) => {
            if block.stmts.iter().any(|stmt| !matches!(stmt.kind, StmtKind::Let(_))) {
                // If there is something else than let bindings, then better not emit the lint.
                return;
            }
            if let Some(block_expr) = block.expr
                // First we ensure that this is a "binding chain" (each statement is a binding
                // of the previous one) and that it is a binding of the closure argument.
                && let Some(last_chain_binding_id) =
                    get_last_chain_binding_hir_id(first_param, block.stmts)
            {
                handle_expr(
                    cx,
                    block_expr,
                    last_chain_binding_id,
                    span,
                    before_chars,
                    revert,
                    is_all,
                );
            }
        },
        ExprKind::Unary(UnOp::Not, expr) => handle_expr(cx, expr, first_param, span, before_chars, !revert, is_all),
        ExprKind::Call(fn_path, [arg]) => {
            // If we have `!is_ascii`, then only `.any()` should warn. And if the condition is
            // `is_ascii`, then only `.all()` should warn.
            if revert != is_all
                && is_path_diagnostic_item(cx, fn_path, sym::char_is_ascii)
                && path_to_local_id(peels_expr_ref(arg), first_param)
                && let Some(snippet) = before_chars.get_source_text(cx)
            {
                span_lint_and_sugg(
                    cx,
                    NEEDLESS_CHARACTER_ITERATION,
                    span,
                    "checking if a string is ascii using iterators",
                    "try",
                    format!("{}{snippet}.is_ascii()", if revert { "!" } else { "" }),
                    Applicability::MachineApplicable,
                );
            }
        },
        _ => {},
    }
}

pub(super) fn check(cx: &LateContext<'_>, call_expr: &Expr<'_>, recv: &Expr<'_>, closure_arg: &Expr<'_>, is_all: bool) {
    if let ExprKind::Closure(&Closure { body, .. }) = closure_arg.kind
        && let body = cx.tcx.hir_body(body)
        && let Some(first_param) = body.params.first()
        && let ExprKind::MethodCall(method, mut recv, [], _) = recv.kind
        && method.ident.name == sym::chars
        && let str_ty = cx.typeck_results().expr_ty_adjusted(recv).peel_refs()
        && *str_ty.kind() == ty::Str
    {
        let expr_start = recv.span;
        while let ExprKind::MethodCall(_, new_recv, _, _) = recv.kind {
            recv = new_recv;
        }
        let body_expr = peel_blocks(body.value);

        handle_expr(
            cx,
            body_expr,
            first_param.pat.hir_id,
            recv.span.with_hi(call_expr.span.hi()),
            recv.span.with_hi(expr_start.hi()),
            false,
            is_all,
        );
    }
}
