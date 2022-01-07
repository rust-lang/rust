use clippy_utils::consts::{self, Constant};
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_applicability;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind, UnOp};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for multiplication by -1 as a form of negation.
    ///
    /// ### Why is this bad?
    /// It's more readable to just negate.
    ///
    /// ### Known problems
    /// This only catches integers (for now).
    ///
    /// ### Example
    /// ```ignore
    /// // Bad
    /// let a = x * -1;
    ///
    /// // Good
    /// let b = -x;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub NEG_MULTIPLY,
    style,
    "multiplying integers by `-1`"
}

declare_lint_pass!(NegMultiply => [NEG_MULTIPLY]);

#[allow(clippy::match_same_arms)]
impl<'tcx> LateLintPass<'tcx> for NegMultiply {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        if let ExprKind::Binary(ref op, left, right) = e.kind {
            if BinOpKind::Mul == op.node {
                match (&left.kind, &right.kind) {
                    (&ExprKind::Unary(..), &ExprKind::Unary(..)) => {},
                    (&ExprKind::Unary(UnOp::Neg, lit), _) => check_mul(cx, e.span, lit, right),
                    (_, &ExprKind::Unary(UnOp::Neg, lit)) => check_mul(cx, e.span, lit, left),
                    _ => {},
                }
            }
        }
    }
}

fn check_mul(cx: &LateContext<'_>, span: Span, lit: &Expr<'_>, exp: &Expr<'_>) {
    if_chain! {
        if let ExprKind::Lit(ref l) = lit.kind;
        if consts::lit_to_constant(&l.node, cx.typeck_results().expr_ty_opt(lit)) == Constant::Int(1);
        if cx.typeck_results().expr_ty(exp).is_integral();

        then {
            let mut applicability = Applicability::MachineApplicable;
            let suggestion = format!("-{}", snippet_with_applicability(cx, exp.span, "..", &mut applicability));
            span_lint_and_sugg(
                    cx,
                    NEG_MULTIPLY,
                    span,
                    "this multiplication by -1 can be written more succinctly",
                    "consider using",
                    suggestion,
                    applicability,
                );
        }
    }
}
