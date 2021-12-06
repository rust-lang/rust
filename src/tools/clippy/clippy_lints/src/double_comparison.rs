//! Lint on unnecessary double comparisons. Some examples:

use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::eq_expr_value;
use clippy_utils::source::snippet_with_applicability;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for double comparisons that could be simplified to a single expression.
    ///
    ///
    /// ### Why is this bad?
    /// Readability.
    ///
    /// ### Example
    /// ```rust
    /// # let x = 1;
    /// # let y = 2;
    /// if x == y || x < y {}
    /// ```
    ///
    /// Could be written as:
    ///
    /// ```rust
    /// # let x = 1;
    /// # let y = 2;
    /// if x <= y {}
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub DOUBLE_COMPARISONS,
    complexity,
    "unnecessary double comparisons that can be simplified"
}

declare_lint_pass!(DoubleComparisons => [DOUBLE_COMPARISONS]);

impl<'tcx> DoubleComparisons {
    #[allow(clippy::similar_names)]
    fn check_binop(cx: &LateContext<'tcx>, op: BinOpKind, lhs: &'tcx Expr<'_>, rhs: &'tcx Expr<'_>, span: Span) {
        let (lkind, llhs, lrhs, rkind, rlhs, rrhs) = match (&lhs.kind, &rhs.kind) {
            (ExprKind::Binary(lb, llhs, lrhs), ExprKind::Binary(rb, rlhs, rrhs)) => {
                (lb.node, llhs, lrhs, rb.node, rlhs, rrhs)
            },
            _ => return,
        };
        if !(eq_expr_value(cx, llhs, rlhs) && eq_expr_value(cx, lrhs, rrhs)) {
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
                    "this binary expression can be simplified",
                    "try",
                    sugg,
                    applicability,
                );
            }};
        }
        #[rustfmt::skip]
        match (op, lkind, rkind) {
            (BinOpKind::Or, BinOpKind::Eq, BinOpKind::Lt) | (BinOpKind::Or, BinOpKind::Lt, BinOpKind::Eq) => {
                lint_double_comparison!(<=);
            },
            (BinOpKind::Or, BinOpKind::Eq, BinOpKind::Gt) | (BinOpKind::Or, BinOpKind::Gt, BinOpKind::Eq) => {
                lint_double_comparison!(>=);
            },
            (BinOpKind::Or, BinOpKind::Lt, BinOpKind::Gt) | (BinOpKind::Or, BinOpKind::Gt, BinOpKind::Lt) => {
                lint_double_comparison!(!=);
            },
            (BinOpKind::And, BinOpKind::Le, BinOpKind::Ge) | (BinOpKind::And, BinOpKind::Ge, BinOpKind::Le) => {
                lint_double_comparison!(==);
            },
            _ => (),
        };
    }
}

impl<'tcx> LateLintPass<'tcx> for DoubleComparisons {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if let ExprKind::Binary(ref kind, lhs, rhs) = expr.kind {
            Self::check_binop(cx, kind.node, lhs, rhs, expr.span);
        }
    }
}
