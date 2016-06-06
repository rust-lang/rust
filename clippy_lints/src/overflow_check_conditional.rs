use rustc::lint::*;
use rustc::hir::*;
use utils::span_lint;

/// **What it does:** This lint finds classic underflow / overflow checks.
///
/// **Why is this bad?** Most classic C underflow / overflow checks will fail in Rust. Users can use functions like `overflowing_*` and `wrapping_*` instead.
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
    // a + b < a, a > a + b, a < a - b, a - b > a
    fn check_expr(&mut self, cx: &LateContext, expr: &Expr) {
        if_let_chain! {[
            let Expr_::ExprBinary(ref op, ref first, ref second) = expr.node,
            let Expr_::ExprBinary(ref op2, ref ident1, ref ident2) = first.node,
            let Expr_::ExprPath(_,ref path1) = ident1.node,
            let Expr_::ExprPath(_, ref path2) = ident2.node,
            let Expr_::ExprPath(_, ref path3) = second.node,
            &path1.segments[0] == &path3.segments[0] || &path2.segments[0] == &path3.segments[0],
            cx.tcx.expr_ty(ident1).is_integral(),
            cx.tcx.expr_ty(ident2).is_integral()
        ], {
            if let BinOp_::BiLt = op.node {
                if let BinOp_::BiAdd = op2.node {
                    span_lint(cx, OVERFLOW_CHECK_CONDITIONAL, expr.span, "You are trying to use classic C overflow conditions that will fail in Rust.");
                }
            }
            if let BinOp_::BiGt = op.node {
                if let BinOp_::BiSub = op2.node {
                    span_lint(cx, OVERFLOW_CHECK_CONDITIONAL, expr.span, "You are trying to use classic C underflow conditions that will fail in Rust.");
                }
            }
        }}

        if_let_chain! {[
            let Expr_::ExprBinary(ref op, ref first, ref second) = expr.node,
            let Expr_::ExprBinary(ref op2, ref ident1, ref ident2) = second.node,
            let Expr_::ExprPath(_,ref path1) = ident1.node,
            let Expr_::ExprPath(_, ref path2) = ident2.node,
            let Expr_::ExprPath(_, ref path3) = first.node,
            &path1.segments[0] == &path3.segments[0] || &path2.segments[0] == &path3.segments[0],
            cx.tcx.expr_ty(ident1).is_integral(),
            cx.tcx.expr_ty(ident2).is_integral()
        ], {
            if let BinOp_::BiGt = op.node {
                if let BinOp_::BiAdd = op2.node {
                    span_lint(cx, OVERFLOW_CHECK_CONDITIONAL, expr.span, "You are trying to use classic C overflow conditions that will fail in Rust.");
                }
            }
            if let BinOp_::BiLt = op.node {
                if let BinOp_::BiSub = op2.node {
                    span_lint(cx, OVERFLOW_CHECK_CONDITIONAL, expr.span, "You are trying to use classic C underflow conditions that will fail in Rust.");
                }
            }
        }}
    }
}
