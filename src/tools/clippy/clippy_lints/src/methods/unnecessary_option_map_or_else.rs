use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::res::MaybeDef;
use clippy_utils::source::snippet_with_applicability;
use clippy_utils::{expr_or_init, find_binding_init, peel_blocks};
use rustc_errors::Applicability;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::{Body, BodyId, Closure, Expr, ExprKind, HirId, QPath};
use rustc_lint::LateContext;
use rustc_span::symbol::sym;

use super::UNNECESSARY_OPTION_MAP_OR_ELSE;
use super::utils::get_last_chain_binding_hir_id;

fn emit_lint(cx: &LateContext<'_>, expr: &Expr<'_>, recv: &Expr<'_>, def_arg: &Expr<'_>) {
    let msg = "unused \"map closure\" when calling `Option::map_or_else` value";
    let mut applicability = Applicability::MachineApplicable;
    let self_snippet = snippet_with_applicability(cx, recv.span, "_", &mut applicability);
    let err_snippet = snippet_with_applicability(cx, def_arg.span, "..", &mut applicability);
    span_lint_and_sugg(
        cx,
        UNNECESSARY_OPTION_MAP_OR_ELSE,
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
        && let Res::Local(hir_id) = path.res
        && expected_hir_id == hir_id
    {
        emit_lint(cx, expr, recv, def_arg);
    }
}

fn handle_closure(cx: &LateContext<'_>, expr: &Expr<'_>, recv: &Expr<'_>, def_arg: &Expr<'_>, body_id: BodyId) {
    let body = cx.tcx.hir_body(body_id);
    handle_fn_body(cx, expr, recv, def_arg, body);
}

fn handle_fn_body(cx: &LateContext<'_>, expr: &Expr<'_>, recv: &Expr<'_>, def_arg: &Expr<'_>, body: &Body<'_>) {
    if let Some(first_param) = body.params.first() {
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

/// lint use of `_.map_or_else(|err| err, |n| n)` for `Option`s.
pub(super) fn check(cx: &LateContext<'_>, expr: &Expr<'_>, recv: &Expr<'_>, def_arg: &Expr<'_>, map_arg: &Expr<'_>) {
    // lint if the caller of `map_or_else()` is an `Option`
    if !cx.typeck_results().expr_ty(recv).is_diag_item(cx, sym::Option) {
        return;
    }
    match map_arg.kind {
        // If the second argument is a closure, we can check its body.
        ExprKind::Closure(&Closure { body, .. }) => {
            handle_closure(cx, expr, recv, def_arg, body);
        },
        ExprKind::Path(qpath) => {
            let res = cx.qpath_res(&qpath, map_arg.hir_id);
            match res {
                // Case 1: Local variable (could be a closure)
                Res::Local(hir_id) => {
                    if let Some(init_expr) = find_binding_init(cx, hir_id) {
                        let origin = expr_or_init(cx, init_expr);
                        if let ExprKind::Closure(&Closure { body, .. }) = origin.kind {
                            handle_closure(cx, expr, recv, def_arg, body);
                        }
                    }
                },
                // Case 2: Function definition
                Res::Def(DefKind::Fn, def_id) => {
                    if let Some(local_def_id) = def_id.as_local()
                        && let Some(body) = cx.tcx.hir_maybe_body_owned_by(local_def_id)
                    {
                        handle_fn_body(cx, expr, recv, def_arg, body);
                    }
                },
                _ => (),
            }
        },
        _ => (),
    }
}
