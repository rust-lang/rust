use rustc::lint::*;
use rustc_front::hir::*;
use utils::{span_lint};

/// **What it does:** This lint finds classic overflow checks.
///
/// **Why is this bad?** Most classic C overflow checks will fail in Rust. Users can use functions like `overflowing_*` and `wrapping_*` instead.
///
/// **Known problems:** None.
///
/// **Example:** `a + b < a`
declare_lint!(pub OVERFLOW_CHECK_CONDITIONAL, Warn,
              "Using overflow checks which are likely to panic");

#[derive(Copy, Clone)]
pub struct OverflowCheckConditional;

impl LintPass for OverflowCheckConditional {
    fn get_lints(&self) -> LintArray {
        lint_array!(OVERFLOW_CHECK_CONDITIONAL)
    }
}

impl LateLintPass for OverflowCheckConditional {
    fn check_expr(&mut self, cx: &LateContext, expr: &Expr) {
        if_let_chain! {[
        let Expr_::ExprBinary(ref op, ref first, ref second) = expr.node,
        let BinOp_::BiLt = op.node,
        let Expr_::ExprBinary(ref op2, ref add1, ref add2) = first.node,
        let BinOp_::BiAdd = op2.node,
        let Expr_::ExprPath(_,ref path1) = add1.node,
        let Expr_::ExprPath(_, ref path2) = add2.node,
        let Expr_::ExprPath(_, ref path3) = second.node,
        (&path1.segments[0]).identifier == (&path3.segments[0]).identifier || (&path2.segments[0]).identifier == (&path3.segments[0]).identifier,
        cx.tcx.expr_ty(add1).is_integral(),
        cx.tcx.expr_ty(add2).is_integral()
        ], {
            span_lint(cx, OVERFLOW_CHECK_CONDITIONAL, expr.span, "You are trying to use classic C overflow conditons that will fail in Rust.");
        }}
    }
}
