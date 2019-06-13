//! Lint on unnecessary double comparisons. Some examples:

use rustc::hir::*;
use rustc::lint::{LateContext, LateLintPass, LintArray, LintPass};
use rustc::{declare_lint_pass, declare_tool_lint};
use rustc_errors::Applicability;
use syntax::source_map::Span;

use crate::utils::{snippet_with_applicability, span_lint_and_sugg, SpanlessEq};

declare_clippy_lint! {
    /// **What it does:** Checks for double comparisons that could be simplified to a single expression.
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
    pub DOUBLE_COMPARISONS,
    complexity,
    "unnecessary double comparisons that can be simplified"
}

declare_lint_pass!(DoubleComparisons => [DOUBLE_COMPARISONS]);

impl<'a, 'tcx> DoubleComparisons {
    #[allow(clippy::similar_names)]
    fn check_binop(self, cx: &LateContext<'a, 'tcx>, op: BinOpKind, lhs: &'tcx Expr, rhs: &'tcx Expr, span: Span) {
        let (lkind, llhs, lrhs, rkind, rlhs, rrhs) = match (lhs.node.clone(), rhs.node.clone()) {
            (ExprKind::Binary(lb, llhs, lrhs), ExprKind::Binary(rb, rlhs, rrhs)) => {
                (lb.node, llhs, lrhs, rb.node, rlhs, rrhs)
            },
            _ => return,
        };
        let mut spanless_eq = SpanlessEq::new(cx).ignore_fn();
        if !(spanless_eq.eq_expr(&llhs, &rlhs) && spanless_eq.eq_expr(&lrhs, &rrhs)) {
            return;
        }
        macro_rules! lint_double_comparison {
            ($op:tt) => {{
                let mut applicability = Applicability::MachineApplicable;
                let lhs_str = snippet_with_applicability(cx, llhs.span, "", &mut applicability);
                let rhs_str = snippet_with_applicability(cx, lrhs.span, "", &mut applicability);
                let sugg = format!("{} {} {}", lhs_str, stringify!($op), rhs_str);
                span_lint_and_sugg(
                    cx,
                    DOUBLE_COMPARISONS,
                    span,
                    "This binary expression can be simplified",
                    "try",
                    sugg,
                    applicability,
                );
            }};
        }
        match (op, lkind, rkind) {
            (BinOpKind::Or, BinOpKind::Eq, BinOpKind::Lt) | (BinOpKind::Or, BinOpKind::Lt, BinOpKind::Eq) => {
                lint_double_comparison!(<=)
            },
            (BinOpKind::Or, BinOpKind::Eq, BinOpKind::Gt) | (BinOpKind::Or, BinOpKind::Gt, BinOpKind::Eq) => {
                lint_double_comparison!(>=)
            },
            (BinOpKind::Or, BinOpKind::Lt, BinOpKind::Gt) | (BinOpKind::Or, BinOpKind::Gt, BinOpKind::Lt) => {
                lint_double_comparison!(!=)
            },
            (BinOpKind::And, BinOpKind::Le, BinOpKind::Ge) | (BinOpKind::And, BinOpKind::Ge, BinOpKind::Le) => {
                lint_double_comparison!(==)
            },
            _ => (),
        };
    }
}

impl<'a, 'tcx> LateLintPass<'a, 'tcx> for DoubleComparisons {
    fn check_expr(&mut self, cx: &LateContext<'a, 'tcx>, expr: &'tcx Expr) {
        if let ExprKind::Binary(ref kind, ref lhs, ref rhs) = expr.node {
            self.check_binop(cx, kind.node, lhs, rhs, expr.span);
        }
    }
}
