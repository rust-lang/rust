use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::peel_blocks;
use clippy_utils::source::snippet;
use clippy_utils::ty::is_type_diagnostic_item;
use rustc_errors::Applicability;
use rustc_hir as hir;
use rustc_hir::{Closure, Expr, ExprKind, HirId, QPath};
use rustc_lint::LateContext;
use rustc_span::symbol::sym;

use super::UNNECESSARY_RESULT_MAP_OR_ELSE;
use super::utils::get_last_chain_binding_hir_id;

fn emit_lint(cx: &LateContext<'_>, expr: &Expr<'_>, recv: &Expr<'_>, def_arg: &Expr<'_>) {
    let msg = "unused \"map closure\" when calling `Result::map_or_else` value";
    let self_snippet = snippet(cx, recv.span, "..");
    let err_snippet = snippet(cx, def_arg.span, "..");
    span_lint_and_sugg(
        cx,
        UNNECESSARY_RESULT_MAP_OR_ELSE,
        expr.span,
        msg,
        "consider using `unwrap_or_else`",
        format!("{self_snippet}.unwrap_or_else({err_snippet})"),
        Applicability::MachineApplicable,
    );
}

fn handle_qpath(
    cx: &LateContext<'_>,
    expr: &Expr<'_>,
    recv: &Expr<'_>,
    def_arg: &Expr<'_>,
    expected_hir_id: HirId,
    qpath: QPath<'_>,
) {
    if let QPath::Resolved(_, path) = qpath
        && let hir::def::Res::Local(hir_id) = path.res
        && expected_hir_id == hir_id
    {
        emit_lint(cx, expr, recv, def_arg);
    }
}

/// lint use of `_.map_or_else(|err| err, |n| n)` for `Result`s.
pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    recv: &'tcx Expr<'_>,
    def_arg: &'tcx Expr<'_>,
    map_arg: &'tcx Expr<'_>,
) {
    // lint if the caller of `map_or_else()` is a `Result`
    if is_type_diagnostic_item(cx, cx.typeck_results().expr_ty(recv), sym::Result)
        && let ExprKind::Closure(&Closure { body, .. }) = map_arg.kind
        && let body = cx.tcx.hir_body(body)
        && let Some(first_param) = body.params.first()
    {
        let body_expr = peel_blocks(body.value);

        match body_expr.kind {
            ExprKind::Path(qpath) => {
                handle_qpath(cx, expr, recv, def_arg, first_param.pat.hir_id, qpath);
            },
            // If this is a block (that wasn't peeled off), then it means there are statements.
            ExprKind::Block(block, _) => {
                if let Some(block_expr) = block.expr
                    // First we ensure that this is a "binding chain" (each statement is a binding
                    // of the previous one) and that it is a binding of the closure argument.
                    && let Some(last_chain_binding_id) =
                        get_last_chain_binding_hir_id(first_param.pat.hir_id, block.stmts)
                    && let ExprKind::Path(qpath) = block_expr.kind
                {
                    handle_qpath(cx, expr, recv, def_arg, last_chain_binding_id, qpath);
                }
            },
            _ => {},
        }
    }
}
