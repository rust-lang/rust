use crate::utils::{get_parent_expr, match_type, paths, span_lint_and_help};
use if_chain::if_chain;
use rustc_hir as hir;
use rustc_lint::LateContext;
use rustc_span::source_map::Span;

use super::FILETYPE_IS_FILE;

pub(super) fn check(cx: &LateContext<'_>, expr: &hir::Expr<'_>, args: &[hir::Expr<'_>]) {
    let ty = cx.typeck_results().expr_ty(&args[0]);

    if !match_type(cx, ty, &paths::FILE_TYPE) {
        return;
    }

    let span: Span;
    let verb: &str;
    let lint_unary: &str;
    let help_unary: &str;
    if_chain! {
        if let Some(parent) = get_parent_expr(cx, expr);
        if let hir::ExprKind::Unary(op, _) = parent.kind;
        if op == hir::UnOp::Not;
        then {
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
    }
    let lint_msg = format!("`{}FileType::is_file()` only {} regular files", lint_unary, verb);
    let help_msg = format!("use `{}FileType::is_dir()` instead", help_unary);
    span_lint_and_help(cx, FILETYPE_IS_FILE, span, &lint_msg, None, &help_msg);
}
