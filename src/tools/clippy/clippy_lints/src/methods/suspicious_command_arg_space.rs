use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::ty::is_type_diagnostic_item;
use rustc_errors::{Applicability, Diag};
use rustc_lint::LateContext;
use rustc_span::{Span, sym};
use {rustc_ast as ast, rustc_hir as hir};

use super::SUSPICIOUS_COMMAND_ARG_SPACE;

pub(super) fn check<'tcx>(cx: &LateContext<'tcx>, recv: &'tcx hir::Expr<'_>, arg: &'tcx hir::Expr<'_>, span: Span) {
    let ty = cx.typeck_results().expr_ty(recv).peel_refs();

    if is_type_diagnostic_item(cx, ty, sym::Command)
        && let hir::ExprKind::Lit(lit) = &arg.kind
        && let ast::LitKind::Str(s, _) = &lit.node
        && let Some((arg1, arg2)) = s.as_str().split_once(' ')
        && arg1.starts_with('-')
        && arg1.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        span_lint_and_then(
            cx,
            SUSPICIOUS_COMMAND_ARG_SPACE,
            arg.span,
            "single argument that looks like it should be multiple arguments",
            |diag: &mut Diag<'_, ()>| {
                diag.multipart_suggestion_verbose(
                    "consider splitting the argument",
                    vec![(span, "args".to_string()), (arg.span, format!("[{arg1:?}, {arg2:?}]"))],
                    Applicability::MaybeIncorrect,
                );
            },
        );
    }
}
