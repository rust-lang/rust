use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::get_parent_expr;
use clippy_utils::ty::is_type_diagnostic_item;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::{Span, sym};

use super::FILETYPE_IS_FILE;

pub(super) fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, recv: &hir::Expr<'_>) {
    let ty = cx.typeck_results().expr_ty(recv);

    if !is_type_diagnostic_item(cx, ty, sym::FileType) {
        return;
    }

    let span: Span;
    let verb: &str;
    let lint_unary: &str;
    let help_unary: &str;
    if let Some(parent) = get_parent_expr(cx, expr)
        && let hir::ExprKind::Unary(op, _) = parent.kind
        && op == hir::UnOp::Not
    {
        lint_unary = "!";
        verb = "denies";
        help_unary = "";
        span = parent.span;
    } else {
        lint_unary = "";
        verb = "covers";
        help_unary = "!";
        span = expr.span;
    }
    let lint_msg = format!("`{lint_unary}FileType::is_file()` only {verb} regular files");

    #[expect(clippy::collapsible_span_lint_calls, reason = "rust-clippy#7797")]
    span_lint_and_then(cx, FILETYPE_IS_FILE, span, lint_msg, |diag| {
        diag.help(format!("use `{help_unary}FileType::is_dir()` instead"));
    });
}
