//! Lint on unnecessary double comparisons. Some examples:

use rustc::hir::*;
use rustc::lint::*;
use syntax::codemap::Span;

use utils::{snippet, span_lint_and_sugg, SpanlessEq};

/// **What it does:** Checks for double comparions that could be simpified to a single expression.
///
///
/// **Why is this bad?** Readability.
///
/// **Known problems:** None.
///
/// **Example:**
/// ```rust
/// x == y || x < y
/// ```
///
/// Could be written as:
///
/// ```rust
/// x <= y
/// ```
declare_lint! {
    pub DOUBLE_COMPARISONS,
    Deny,
    "unnecessary double comparisons that can be simplified"
}

pub struct DoubleComparisonPass;

impl LintPass for DoubleComparisonPass {
    fn get_lints(&self) -> LintArray {
        lint_array!(DOUBLE_COMPARISONS)
    }
}

impl<'a, 'tcx> DoubleComparisonPass {
    fn check_binop(
        &self,
        cx: &LateContext<'a, 'tcx>,
        op: BinOp_,
        lhs: &'tcx Expr,
        rhs: &'tcx Expr,
        span: Span,
    ) {
        let (lkind, llhs, lrhs, rkind, rlhs, rrhs) = match (lhs.node.clone(), rhs.node.clone()) {
            (ExprBinary(lb, llhs, lrhs), ExprBinary(rb, rlhs, rrhs)) => {
                (lb.node, llhs, lrhs, rb.node, rlhs, rrhs)
            }
            _ => return,
        };
        let spanless_eq = SpanlessEq::new(cx).ignore_fn();
        if !(spanless_eq.eq_expr(&llhs, &rlhs) && spanless_eq.eq_expr(&lrhs, &rrhs)) {
            return;
        }
        macro_rules! lint_double_comparison {
            ($op:tt) => {{
                let lhs_str = snippet(cx, llhs.span, "");
                let rhs_str = snippet(cx, lrhs.span, "");
                let sugg = format!("{} {} {}", lhs_str, stringify!($op), rhs_str);
                span_lint_and_sugg(cx, DOUBLE_COMPARISONS, span,
                                   "This binary expression can be simplified",
                                   "try", sugg);
            }}
        }
        match (op, lkind, rkind) {
            (BiOr, BiEq, BiLt) | (BiOr, BiLt, BiEq) => lint_double_comparison!(<=),
            (BiOr, BiEq, BiGt) | (BiOr, BiGt, BiEq) => lint_double_comparison!(>=),
            (BiOr, BiLt, BiGt) | (BiOr, BiGt, BiLt) => lint_double_comparison!(!=),
            (BiAnd, BiLe, BiGe) | (BiAnd, BiGe, BiLe) => lint_double_comparison!(==),
            _ => (),
        };
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for DoubleComparisonPass {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if let ExprBinary(ref kind, ref lhs, ref rhs) = expr.node {
            self.check_binop(cx, kind.node, lhs, rhs, expr.span);
        }
    }
}
