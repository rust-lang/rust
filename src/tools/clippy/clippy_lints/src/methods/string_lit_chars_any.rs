use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::msrvs::{self, Msrv};
use clippy_utils::source::SpanRangeExt;
use clippy_utils::{is_from_proc_macro, is_trait_method, path_to_local};
use itertools::Itertools;
use rustc_ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind, Param, PatKind};
use rustc_lint::LateContext;
use rustc_span::sym;

use super::STRING_LIT_CHARS_ANY;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'tcx>,
    recv: &Expr<'_>,
    param: &'tcx Param<'tcx>,
    body: &Expr<'_>,
    msrv: Msrv,
) {
    if is_trait_method(cx, expr, sym::Iterator)
        && let PatKind::Binding(_, arg, _, _) = param.pat.kind
        && let ExprKind::Lit(lit_kind) = recv.kind
        && let LitKind::Str(val, _) = lit_kind.node
        && let ExprKind::Binary(kind, lhs, rhs) = body.kind
        && let BinOpKind::Eq = kind.node
        && let Some(lhs_path) = path_to_local(lhs)
        && let Some(rhs_path) = path_to_local(rhs)
        && let scrutinee = match (lhs_path == arg, rhs_path == arg) {
            (true, false) => rhs,
            (false, true) => lhs,
            _ => return,
        }
        && msrv.meets(cx, msrvs::MATCHES_MACRO)
        && !is_from_proc_macro(cx, expr)
        && let Some(scrutinee_snip) = scrutinee.span.get_source_text(cx)
    {
        // Normalize the char using `map` so `join` doesn't use `Display`, if we don't then
        // something like `r"\"` will become `'\'`, which is of course invalid
        let pat_snip = val.as_str().chars().map(|c| format!("{c:?}")).join(" | ");

        span_lint_and_then(
            cx,
            STRING_LIT_CHARS_ANY,
            expr.span,
            "usage of `.chars().any(...)` to check if a char matches any from a string literal",
            |diag| {
                diag.span_suggestion_verbose(
                    expr.span,
                    "use `matches!(...)` instead",
                    format!("matches!({scrutinee_snip}, {pat_snip})"),
                    Applicability::MachineApplicable,
                );
            },
        );
    }
}
