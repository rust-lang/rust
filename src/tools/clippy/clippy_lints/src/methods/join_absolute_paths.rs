use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::expr_or_init;
use clippy_utils::source::snippet;
use clippy_utils::ty::is_type_diagnostic_item;
use rustc_ast::ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_span::Span;
use rustc_span::symbol::sym;

use super::JOIN_ABSOLUTE_PATHS;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, recv: &'tcx Expr<'tcx>, join_arg: &'tcx Expr<'tcx>, expr_span: Span) {
    let ty = cx.typeck_results().expr_ty(recv).peel_refs();
    if (is_type_diagnostic_item(cx, ty, sym::Path) || is_type_diagnostic_item(cx, ty, sym::PathBuf))
        && let ExprKind::Lit(spanned) = expr_or_init(cx, join_arg).kind
        && let LitKind::Str(symbol, _) = spanned.node
        && let sym_str = symbol.as_str()
        && sym_str.starts_with(['/', '\\'])
    {
        span_lint_and_then(
            cx,
            JOIN_ABSOLUTE_PATHS,
            join_arg.span,
            "argument to `Path::join` starts with a path separator",
            |diag| {
                let arg_str = snippet(cx, spanned.span, "..");

                let no_separator = if sym_str.starts_with('/') {
                    arg_str.replacen('/', "", 1)
                } else {
                    arg_str.replacen('\\', "", 1)
                };

                diag.note("joining a path starting with separator will replace the path instead")
                    .span_suggestion(
                        spanned.span,
                        "if this is unintentional, try removing the starting separator",
                        no_separator,
                        Applicability::Unspecified,
                    )
                    .span_suggestion(
                        expr_span,
                        "if this is intentional, consider using `Path::new`",
                        format!("PathBuf::from({arg_str})"),
                        Applicability::Unspecified,
                    );
            },
        );
    }
}
