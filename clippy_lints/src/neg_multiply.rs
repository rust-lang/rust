use if_chain::if_chain;
use rustc::hir::*;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};
use syntax::source_map::{Span, Spanned};

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
    "multiplying integers with -1"
}

declare_lint_pass!(NegMultiply => [NEG_MULTIPLY]);

#[allow(clippy::match_same_arms)]
impl<'a, 'tcx> LateLintPass<'a, 'tcx> for NegMultiply {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, e: &'tcx Expr) {
        if let ExprKind::Binary(
            Spanned {
                node: BinOpKind::Mul, ..
            },
            ref l,
            ref r,
        ) = e.node
        {
            match (&l.node, &r.node) {
                (&ExprKind::Unary(..), &ExprKind::Unary(..)) => (),
                (&ExprKind::Unary(UnNeg, ref lit), _) => check_mul(cx, e.span, lit, r),
                (_, &ExprKind::Unary(UnNeg, ref lit)) => check_mul(cx, e.span, lit, l),
                _ => (),
            }
        }
    }
}

fn check_mul(cx: &LateContext<'_, '_>, span: Span, lit: &Expr, exp: &Expr) {
    if_chain! {
        if let ExprKind::Lit(ref l) = lit.node;
        if let Constant::Int(val) = consts::lit_to_constant(&l.node, cx.tables.expr_ty(lit));
        if val == 1;
        if cx.tables.expr_ty(exp).is_integral();
        then {
            span_lint(cx,
                      NEG_MULTIPLY,
                      span,
                      "Negation by multiplying with -1");
        }
    }
}
