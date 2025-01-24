use clippy_utils::consts::{ConstEvalCtxt, Constant};
use clippy_utils::diagnostics::span_lint;
use clippy_utils::ty::same_type_and_consts;

use rustc_hir::{BinOpKind, Expr};
use rustc_lint::LateContext;
use rustc_middle::ty::TypeckResults;

use super::ERASING_OP;

pub(super) fn check<'tcx>(
    cx: &LateContext<'tcx>,
    e: &'tcx Expr<'_>,
    op: BinOpKind,
    left: &'tcx Expr<'_>,
    right: &'tcx Expr<'_>,
) {
    let tck = cx.typeck_results();
    match op {
        BinOpKind::Mul | BinOpKind::BitAnd => {
            check_op(cx, tck, left, right, e);
            check_op(cx, tck, right, left, e);
        },
        BinOpKind::Div => check_op(cx, tck, left, right, e),
        _ => (),
    }
}

fn different_types(tck: &TypeckResults<'_>, input: &Expr<'_>, output: &Expr<'_>) -> bool {
    let input_ty = tck.expr_ty(input).peel_refs();
    let output_ty = tck.expr_ty(output).peel_refs();
    !same_type_and_consts(input_ty, output_ty)
}

fn check_op<'tcx>(
    cx: &LateContext<'tcx>,
    tck: &'tcx TypeckResults<'tcx>,
    op: &Expr<'tcx>,
    other: &Expr<'tcx>,
    parent: &Expr<'tcx>,
) {
    if ConstEvalCtxt::with_env(cx.tcx, cx.typing_env(), tck).eval_simple(op) == Some(Constant::Int(0)) {
        if different_types(tck, other, parent) {
            return;
        }
        span_lint(
            cx,
            ERASING_OP,
            parent.span,
            "this operation will always return zero. This is likely not the intended outcome",
        );
    }
}
