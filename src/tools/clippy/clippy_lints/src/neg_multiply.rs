use if_chain::if_chain;
use rustc_hir::{BinOpKind, Expr, ExprKind, UnOp};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;

use crate::consts::{self, Constant};
use crate::utils::span_lint;

declare_clippy_lint! {
    /// **What it does:** Checks for multiplication by -1 as a form of negation.
    ///
    /// **Why is this bad?** It's more readable to just negate.
    ///
    /// **Known problems:** This only catches integers (for now).
    ///
    /// **Example:**
    /// ```ignore
    /// x * -1
    /// ```
    pub NEG_MULTIPLY,
    style,
    "multiplying integers with `-1`"
}

declare_lint_pass!(NegMultiply => [NEG_MULTIPLY]);

#[allow(clippy::match_same_arms)]
impl<'tcx> LateLintPass<'tcx> for NegMultiply {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        if let ExprKind::Binary(ref op, ref left, ref right) = e.kind {
            if BinOpKind::Mul == op.node {
                match (&left.kind, &right.kind) {
                    (&ExprKind::Unary(..), &ExprKind::Unary(..)) => {},
                    (&ExprKind::Unary(UnOp::Neg, ref lit), _) => check_mul(cx, e.span, lit, right),
                    (_, &ExprKind::Unary(UnOp::Neg, ref lit)) => check_mul(cx, e.span, lit, left),
                    _ => {},
                }
            }
        }
    }
}

fn check_mul(cx: &LateContext<'_>, span: Span, lit: &Expr<'_>, exp: &Expr<'_>) {
    if_chain! {
        if let ExprKind::Lit(ref l) = lit.kind;
        if let Constant::Int(1) = consts::lit_to_constant(&l.node, cx.typeck_results().expr_ty_opt(lit));
        if cx.typeck_results().expr_ty(exp).is_integral();
        then {
            span_lint(cx, NEG_MULTIPLY, span, "negation by multiplying with `-1`");
        }
    }
}
