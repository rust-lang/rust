use rustc::hir::*;
use rustc::lint::*;
use syntax::codemap::{Span, Spanned};

use consts::{self, Constant};
use utils::span_lint;

/// **What it does:** Checks for multiplication by -1 as a form of negation.
///
/// **Why is this bad?** It's more readable to just negate.
///
/// **Known problems:** This only catches integers (for now).
///
/// **Example:**
/// ```rust
/// x * -1
/// ```
declare_lint! {
    pub NEG_MULTIPLY,
    Warn,
    "multiplying integers with -1"
}

#[derive(Copy, Clone)]
pub struct NegMultiply;

impl LintPass for NegMultiply {
    fn get_lints(&self) -> LintArray {
        lint_array!(NEG_MULTIPLY)
    }
}

#[allow(match_same_arms)]
impl<'a, 'tcx> LateLintPass<'a, 'tcx> for NegMultiply {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, e: &'tcx Expr) {
        if let ExprBinary(Spanned { node: BiMul, .. }, ref l, ref r) = e.node {
            match (&l.node, &r.node) {
                (&ExprUnary(..), &ExprUnary(..)) => (),
                (&ExprUnary(UnNeg, ref lit), _) => check_mul(cx, e.span, lit, r),
                (_, &ExprUnary(UnNeg, ref lit)) => check_mul(cx, e.span, lit, l),
                _ => (),
            }
        }
    }
}

fn check_mul(cx: &LateContext, span: Span, lit: &Expr, exp: &Expr) {
    if_chain! {
        if let ExprLit(ref l) = lit.node;
        if let Constant::Int(ref ci) = consts::lit_to_constant(&l.node, cx.tcx, cx.tables.expr_ty(lit));
        if let Some(val) = ci.to_u64();
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
