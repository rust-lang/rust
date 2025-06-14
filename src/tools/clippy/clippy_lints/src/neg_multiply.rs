use clippy_utils::consts::{self, Constant};
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_context;
use clippy_utils::sugg::has_enclosing_paren;
use rustc_ast::util::parser::ExprPrecedence;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind, UnOp};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::Span;

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

impl<'tcx> LateLintPass<'tcx> for NegMultiply {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, e: &'tcx Expr<'_>) {
        if let ExprKind::Binary(ref op, left, right) = e.kind
            && BinOpKind::Mul == op.node
        {
            match (&left.kind, &right.kind) {
                (&ExprKind::Unary(..), &ExprKind::Unary(..)) => {},
                (&ExprKind::Unary(UnOp::Neg, lit), _) => check_mul(cx, e.span, lit, right),
                (_, &ExprKind::Unary(UnOp::Neg, lit)) => check_mul(cx, e.span, lit, left),
                _ => {},
            }
        }
    }
}

fn check_mul(cx: &LateContext<'_>, span: Span, lit: &Expr<'_>, exp: &Expr<'_>) {
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
        let (snip, from_macro) = snippet_with_context(cx, exp.span, span.ctxt(), "..", &mut applicability);
        let suggestion = if !from_macro && cx.precedence(exp) < ExprPrecedence::Prefix && !has_enclosing_paren(&snip) {
            format!("-({snip})")
        } else {
            format!("-{snip}")
        };
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
