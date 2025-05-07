use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet;
use clippy_utils::ty::is_type_diagnostic_item;
use clippy_utils::{is_res_lang_ctor, path_res, peel_blocks};
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::LangItem::{OptionNone, OptionSome};
use rustc_lint::LateContext;
use rustc_span::symbol::sym;

use super::RESULT_MAP_OR_INTO_OPTION;

/// lint use of `_.map_or_else(|_| None, Some)` for `Result`s
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx hir::Expr<'_>,
    recv: &'tcx hir::Expr<'_>,
    def_arg: &'tcx hir::Expr<'_>,
    map_arg: &'tcx hir::Expr<'_>,
) {
    // lint if the caller of `map_or_else()` is a `Result`
    if is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(recv), sym::Result)
        // We check that it is mapped as `Some`.
        && is_res_lang_ctor(cx, path_res(cx, map_arg), OptionSome)
        && let hir::ExprKind::Closure(&hir::Closure { body, .. }) = def_arg.kind
        && let body = cx.tcx.hir_body(body)
        // And finally we check that we return a `None` in the "else case".
        && is_res_lang_ctor(cx, path_res(cx, peel_blocks(body.value)), OptionNone)
    {
        let msg = "called `map_or_else(|_| None, Some)` on a `Result` value";
        let self_snippet = snippet(cx, recv.span, "..");
        span_lint_and_sugg(
            cx,
            RESULT_MAP_OR_INTO_OPTION,
            expr.span,
            msg,
            "consider using `ok`",
            format!("{self_snippet}.ok()"),
            Applicability::MachineApplicable,
        );
    }
}
