use super::manual_utils::{check_with, SomeExpr};
use super::MANUAL_MAP;
use clippy_utils::diagnostics::span_lint_and_sugg;

use clippy_utils::{is_res_lang_ctor, path_res};

use rustc_hir::LangItem::OptionSome;
use rustc_hir::{Arm, Block, BlockCheckMode, Expr, ExprKind, Pat, UnsafeSource};
use rustc_lint::LateContext;
use rustc_span::SyntaxContext;

pub(super) fn check_match<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    scrutinee: &'tcx Expr<'_>,
    arms: &'tcx [Arm<'_>],
) {
    if let [arm1, arm2] = arms
        && arm1.guard.is_none()
        && arm2.guard.is_none()
    {
        check(cx, expr, scrutinee, arm1.pat, arm1.body, Some(arm2.pat), arm2.body);
    }
}

pub(super) fn check_if_let<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    let_pat: &'tcx Pat<'_>,
    let_expr: &'tcx Expr<'_>,
    then_expr: &'tcx Expr<'_>,
    else_expr: &'tcx Expr<'_>,
) {
    check(cx, expr, let_expr, let_pat, then_expr, None, else_expr);
}

fn check<'tcx>(
    cx: &LateContext<'tcx>,
    expr: &'tcx Expr<'_>,
    scrutinee: &'tcx Expr<'_>,
    then_pat: &'tcx Pat<'_>,
    then_body: &'tcx Expr<'_>,
    else_pat: Option<&'tcx Pat<'_>>,
    else_body: &'tcx Expr<'_>,
) {
    if let Some(sugg_info) = check_with(
        cx,
        expr,
        scrutinee,
        then_pat,
        then_body,
        else_pat,
        else_body,
        get_some_expr,
    ) {
        span_lint_and_sugg(
            cx,
            MANUAL_MAP,
            expr.span,
            "manual implementation of `Option::map`",
            "try",
            if sugg_info.needs_brackets {
                format!(
                    "{{ {}{}.map({}) }}",
                    sugg_info.scrutinee_str, sugg_info.as_ref_str, sugg_info.body_str
                )
            } else {
                format!(
                    "{}{}.map({})",
                    sugg_info.scrutinee_str, sugg_info.as_ref_str, sugg_info.body_str
                )
            },
            sugg_info.app,
        );
    }
}

// Checks for an expression wrapped by the `Some` constructor. Returns the contained expression.
fn get_some_expr<'tcx>(
    cx: &LateContext<'tcx>,
    _: &'tcx Pat<'_>,
    expr: &'tcx Expr<'_>,
    ctxt: SyntaxContext,
) -> Option<SomeExpr<'tcx>> {
    fn get_some_expr_internal<'tcx>(
        cx: &LateContext<'tcx>,
        expr: &'tcx Expr<'_>,
        needs_unsafe_block: bool,
        ctxt: SyntaxContext,
    ) -> Option<SomeExpr<'tcx>> {
        // TODO: Allow more complex expressions.
        match expr.kind {
            ExprKind::Call(callee, [arg])
                if ctxt == expr.span.ctxt() && is_res_lang_ctor(cx, path_res(cx, callee), OptionSome) =>
            {
                Some(SomeExpr::new_no_negated(arg, needs_unsafe_block))
            },
            ExprKind::Block(
                Block {
                    stmts: [],
                    expr: Some(expr),
                    rules,
                    ..
                },
                _,
            ) => get_some_expr_internal(
                cx,
                expr,
                needs_unsafe_block || *rules == BlockCheckMode::UnsafeBlock(UnsafeSource::UserProvided),
                ctxt,
            ),
            _ => None,
        }
    }
    get_some_expr_internal(cx, expr, false, ctxt)
}
