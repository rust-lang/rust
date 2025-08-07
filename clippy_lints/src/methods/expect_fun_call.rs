use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::macros::{FormatArgsStorage, format_args_inputs_span, root_macro_call_first_node};
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::ty::{is_type_diagnostic_item, is_type_lang_item};
use clippy_utils::visitors::for_each_expr;
use clippy_utils::{contains_return, is_inside_always_const_context, peel_blocks};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::symbol::sym;
use rustc_span::{Span, Symbol};
use std::borrow::Cow;
use std::ops::ControlFlow;

use super::EXPECT_FUN_CALL;

/// Checks for the `EXPECT_FUN_CALL` lint.
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    format_args_storage: &FormatArgsStorage,
    expr: &hir::Expr<'_>,
    method_span: Span,
    name: Symbol,
    receiver: &'tcx hir::Expr<'tcx>,
    args: &'tcx [hir::Expr<'tcx>],
) {
    if name == sym::expect
        && let [arg] = args
        && let arg_root = get_arg_root(cx, arg)
        && contains_call(cx, arg_root)
        && !contains_return(arg_root)
    {
        let receiver_type = cx.typeck_results().expr_ty_adjusted(receiver);
        let closure_args = if is_type_diagnostic_item(cx, receiver_type, sym::Option) {
            "||"
        } else if is_type_diagnostic_item(cx, receiver_type, sym::Result) {
            "|_|"
        } else {
            return;
        };

        let span_replace_word = method_span.with_hi(expr.span.hi());

        let mut applicability = Applicability::MachineApplicable;

        // Special handling for `format!` as arg_root
        if let Some(macro_call) = root_macro_call_first_node(cx, arg_root) {
            if cx.tcx.is_diagnostic_item(sym::format_macro, macro_call.def_id)
                && let Some(format_args) = format_args_storage.get(cx, arg_root, macro_call.expn)
            {
                let span = format_args_inputs_span(format_args);
                let sugg = snippet_with_applicability(cx, span, "..", &mut applicability);
                span_lint_and_sugg(
                    cx,
                    EXPECT_FUN_CALL,
                    span_replace_word,
                    format!("function call inside of `{name}`"),
                    "try",
                    format!("unwrap_or_else({closure_args} panic!({sugg}))"),
                    applicability,
                );
            }
            return;
        }

        let arg_root_snippet: Cow<'_, _> = snippet_with_applicability(cx, arg_root.span, "..", &mut applicability);

        span_lint_and_sugg(
            cx,
            EXPECT_FUN_CALL,
            span_replace_word,
            format!("function call inside of `{name}`"),
            "try",
            format!("unwrap_or_else({closure_args} panic!(\"{{}}\", {arg_root_snippet}))"),
            applicability,
        );
    }
}

/// Strip `{}`, `&`, `as_ref()` and `as_str()` off `arg` until we're left with either a `String` or
/// `&str`
fn get_arg_root<'a>(cx: &LateContext<'_>, arg: &'a hir::Expr<'a>) -> &'a hir::Expr<'a> {
    let mut arg_root = peel_blocks(arg);
    loop {
        arg_root = match &arg_root.kind {
            hir::ExprKind::AddrOf(hir::BorrowKind::Ref, _, expr) => expr,
            hir::ExprKind::MethodCall(method_name, receiver, [], ..) => {
                if (method_name.ident.name == sym::as_str || method_name.ident.name == sym::as_ref) && {
                    let arg_type = cx.typeck_results().expr_ty(receiver);
                    let base_type = arg_type.peel_refs();
                    base_type.is_str() || is_type_lang_item(cx, base_type, hir::LangItem::String)
                } {
                    receiver
                } else {
                    break;
                }
            },
            _ => break,
        };
    }
    arg_root
}

fn contains_call<'a>(cx: &LateContext<'a>, arg: &'a hir::Expr<'a>) -> bool {
    for_each_expr(cx, arg, |expr| {
        if matches!(expr.kind, hir::ExprKind::MethodCall { .. } | hir::ExprKind::Call { .. })
            && !is_inside_always_const_context(cx.tcx, expr.hir_id)
        {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    })
    .is_some()
}
