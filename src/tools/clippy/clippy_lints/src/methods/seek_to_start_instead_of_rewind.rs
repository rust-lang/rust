use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::ty::implements_trait;
use clippy_utils::{get_trait_def_id, is_expr_used_or_unified, match_def_path, paths};
use rustc_ast::ast::{LitIntType, LitKind};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::LateContext;
use rustc_span::Span;

use super::SEEK_TO_START_INSTEAD_OF_REWIND;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    recv: &'tcx Expr<'_>,
    arg: &'tcx Expr<'_>,
    name_span: Span,
) {
    // Get receiver type
    let ty = cx.typeck_results().expr_ty(recv).peel_refs();

    if is_expr_used_or_unified(cx.tcx, expr) {
        return;
    }

    if let Some(seek_trait_id) = get_trait_def_id(cx, &paths::STD_IO_SEEK) &&
        implements_trait(cx, ty, seek_trait_id, &[]) &&
        let ExprKind::Call(func, args1) = arg.kind &&
        let ExprKind::Path(ref path) = func.kind &&
        let Some(def_id) = cx.qpath_res(path, func.hir_id).opt_def_id() &&
        match_def_path(cx, def_id, &paths::STD_IO_SEEKFROM_START) &&
        args1.len() == 1 &&
        let ExprKind::Lit(ref lit) = args1[0].kind &&
        let LitKind::Int(0, LitIntType::Unsuffixed) = lit.node
    {
        let method_call_span = expr.span.with_lo(name_span.lo());
        span_lint_and_then(
            cx,
            SEEK_TO_START_INSTEAD_OF_REWIND,
            method_call_span,
            "used `seek` to go to the start of the stream",
            |diag| {
                let app = Applicability::MachineApplicable;

                diag.span_suggestion(method_call_span, "replace with", "rewind()", app);
            },
        );
    }
}
