use crate::methods::TRIM_SPLIT_WHITESPACE;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::sym;
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_span::Span;
use rustc_span::def_id::DefId;

pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, split_recv: &Expr<'_>, split_ws_span: Span) {
    let tyckres = cx.typeck_results();
    if let Some(split_ws_def_id) = tyckres.type_dependent_def_id(expr.hir_id)
        && cx.tcx.is_diagnostic_item(sym::str_split_whitespace, split_ws_def_id)
        && let ExprKind::MethodCall(path, _trim_recv, [], trim_span) = split_recv.kind
        && let trim_fn_name @ (sym::trim | sym::trim_start | sym::trim_end) = path.ident.name
        && let Some(trim_def_id) = tyckres.type_dependent_def_id(split_recv.hir_id)
        && is_one_of_trim_diagnostic_items(cx, trim_def_id)
    {
        span_lint_and_sugg(
            cx,
            TRIM_SPLIT_WHITESPACE,
            trim_span.with_hi(split_ws_span.lo()),
            format!("found call to `str::{trim_fn_name}` before `str::split_whitespace`"),
            format!("remove `{trim_fn_name}()`"),
            String::new(),
            Applicability::MachineApplicable,
        );
    }
}

fn is_one_of_trim_diagnostic_items(cx: &LateContext<'_>, trim_def_id: DefId) -> bool {
    matches!(
        cx.tcx.get_diagnostic_name(trim_def_id),
        Some(sym::str_trim | sym::str_trim_start | sym::str_trim_end)
    )
}
