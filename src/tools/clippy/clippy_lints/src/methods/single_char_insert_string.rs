use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::{snippet_with_applicability, str_literal_to_char_literal};
use rustc_ast::BorrowKind;
use rustc_errors::Applicability;
use rustc_hir::{self as hir, ExprKind};
use rustc_lint::LateContext;

use super::SINGLE_CHAR_ADD_STR;

/// lint for length-1 `str`s as argument for `insert_str`
pub(super) fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, receiver: &hir::Expr<'_>, args: &[hir::Expr<'_>]) {
    let mut applicability = Applicability::MachineApplicable;
    if let Some(extension_string) = str_literal_to_char_literal(cx, &args[1], &mut applicability, false) {
        let base_string_snippet =
            snippet_with_applicability(cx, receiver.span.source_callsite(), "_", &mut applicability);
        let pos_arg = snippet_with_applicability(cx, args[0].span, "..", &mut applicability);
        let sugg = format!("{base_string_snippet}.insert({pos_arg}, {extension_string})");
        span_lint_and_sugg(
            cx,
            SINGLE_CHAR_ADD_STR,
            expr.span,
            "calling `insert_str()` using a single-character string literal",
            "consider using `insert` with a character literal",
            sugg,
            applicability,
        );
    }

    if let ExprKind::AddrOf(BorrowKind::Ref, _, arg) = &args[1].kind
        && let ExprKind::MethodCall(path_segment, method_arg, [], _) = &arg.kind
        && path_segment.ident.name == rustc_span::sym::to_string
        && (is_ref_char(cx, method_arg) || is_char(cx, method_arg))
    {
        let base_string_snippet =
            snippet_with_applicability(cx, receiver.span.source_callsite(), "..", &mut applicability);
        let extension_string =
            snippet_with_applicability(cx, method_arg.span.source_callsite(), "..", &mut applicability);
        let pos_arg = snippet_with_applicability(cx, args[0].span, "..", &mut applicability);
        let deref_string = if is_ref_char(cx, method_arg) { "*" } else { "" };

        let sugg = format!("{base_string_snippet}.insert({pos_arg}, {deref_string}{extension_string})");
        span_lint_and_sugg(
            cx,
            SINGLE_CHAR_ADD_STR,
            expr.span,
            "calling `insert_str()` using a single-character converted to string",
            "consider using `insert` without `to_string()`",
            sugg,
            applicability,
        );
    }
}

fn is_ref_char(cx: &LateContext<'_>, expr: &hir::Expr<'_>) -> bool {
    if cx.typeck_results().expr_ty(expr).is_ref()
        && let rustc_middle::ty::Ref(_, ty, _) = cx.typeck_results().expr_ty(expr).kind()
        && ty.is_char()
    {
        return true;
    }

    false
}

fn is_char(cx: &LateContext<'_>, expr: &hir::Expr<'_>) -> bool {
    cx.typeck_results().expr_ty(expr).is_char()
}
