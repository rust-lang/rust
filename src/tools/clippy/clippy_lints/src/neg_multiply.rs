use clippy_utils::consts::{self, Constant};
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::get_parent_expr;
use clippy_utils::source::{snippet, snippet_with_context};
use clippy_utils::sugg::has_enclosing_paren;
use rustc_ast::util::parser::ExprPrecedence;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind, UnOp};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for multiplication by -1 as a form of negation.
    ///
    /// ### Why is this bad?
    /// It's more readable to just negate.
    ///
    /// ### Example
    /// ```rust,ignore
    /// let a = x * -1;
    /// ```
    ///
    /// Use instead:
    /// ```rust,ignore
    /// let a = -x;
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub NEG_MULTIPLY,
    style,
    "multiplying integers by `-1`"
}

declare_lint_pass!(NegMultiply => [NEG_MULTIPLY]);

fn is_in_parens_with_postfix(cx: &LateContext<'_>, mul_expr: &Expr<'_>) -> bool {
    if let Some(parent) = get_parent_expr(cx, mul_expr) {
        let mult_snippet = snippet(cx, mul_expr.span, "");
        if has_enclosing_paren(&mult_snippet)
            && let ExprKind::MethodCall(_, _, _, _) = parent.kind
        {
            return true;
        }
    }

    false
}

impl<'tcx> LateLintPass<'tcx> for NegMultiply {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        if let ExprKind::Binary(ref op, left, right) = e.kind
            && BinOpKind::Mul == op.node
        {
            match (&left.kind, &right.kind) {
                (&ExprKind::Unary(..), &ExprKind::Unary(..)) => {},
                (&ExprKind::Unary(UnOp::Neg, lit), _) => check_mul(cx, e, lit, right),
                (_, &ExprKind::Unary(UnOp::Neg, lit)) => check_mul(cx, e, lit, left),
                _ => {},
            }
        }
    }
}

fn check_mul(cx: &LateContext<'_>, mul_expr: &Expr<'_>, lit: &Expr<'_>, exp: &Expr<'_>) {
    const F16_ONE: u16 = 1.0_f16.to_bits();
    const F128_ONE: u128 = 1.0_f128.to_bits();
    if let ExprKind::Lit(l) = lit.kind
        && matches!(
            consts::lit_to_mir_constant(&l.node, cx.typeck_results().expr_ty_opt(lit)),
            Constant::Int(1)
                | Constant::F16(F16_ONE)
                | Constant::F32(1.0)
                | Constant::F64(1.0)
                | Constant::F128(F128_ONE)
        )
        && cx.typeck_results().expr_ty(exp).is_numeric()
    {
        let mut applicability = Applicability::MachineApplicable;
        let (snip, from_macro) = snippet_with_context(cx, exp.span, mul_expr.span.ctxt(), "..", &mut applicability);

        let needs_parens_for_postfix = is_in_parens_with_postfix(cx, mul_expr);

        let suggestion = if needs_parens_for_postfix {
            // Special case: when the multiplication is in parentheses followed by a method call
            // we need to preserve the grouping but negate the inner expression.
            // Consider this expression: `((a.delta - 0.5).abs() * -1.0).total_cmp(&1.0)`
            // We need to end up with: `(-(a.delta - 0.5).abs()).total_cmp(&1.0)`
            // Otherwise, without the parentheses we would try to negate an Ordering:
            // `-(a.delta - 0.5).abs().total_cmp(&1.0)`
            format!("(-{snip})")
        } else if !from_macro && cx.precedence(exp) < ExprPrecedence::Prefix && !has_enclosing_paren(&snip) {
            format!("-({snip})")
        } else {
            format!("-{snip}")
        };
        span_lint_and_sugg(
            cx,
            NEG_MULTIPLY,
            mul_expr.span,
            "this multiplication by -1 can be written more succinctly",
            "consider using",
            suggestion,
            applicability,
        );
    }
}
