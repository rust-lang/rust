use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::peel_blocks;
use clippy_utils::res::{MaybeDef, MaybeQPath};
use clippy_utils::source::snippet;
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
    if cx.typeck_results().expr_ty(recv).is_diag_item(cx, sym::Result)
        // We check that it is mapped as `Some`.
        && map_arg.res(cx).ctor_parent(cx).is_lang_item(cx, OptionSome)
        && let hir::ExprKind::Closure(&hir::Closure { body, .. }) = def_arg.kind
        && let body = cx.tcx.hir_body(body)
        // And finally we check that we return a `None` in the "else case".
        && peel_blocks(body.value).res(cx).ctor_parent(cx).is_lang_item(cx, OptionNone)
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
