use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::SpanExt;
use rustc_ast::LitKind;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lexer::is_whitespace;
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Warns for a Bitwise XOR (`^`) operator being probably confused as a powering. It will not trigger if any of the numbers are not in decimal.
    ///
    /// ### Why restrict this?
    /// It's most probably a typo and may lead to unexpected behaviours.
    ///
    /// ### Example
    /// ```no_run
    /// let x = 3_i32 ^ 4_i32;
    /// ```
    /// Use instead:
    /// ```no_run
    /// let x = 3_i32.pow(4);
    /// ```
    #[clippy::version = "1.67.0"]
    pub SUSPICIOUS_XOR_USED_AS_POW,
    restriction,
    "XOR (`^`) operator possibly used as exponentiation operator"
}

declare_lint_pass!(ConfusingXorAndPow => [SUSPICIOUS_XOR_USED_AS_POW]);

impl LateLintPass<'_> for ConfusingXorAndPow {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        if let ExprKind::Binary(op, left, right) = &expr.kind
            && op.node == BinOpKind::BitXor
            && let ExprKind::Lit(lit_left) = &left.kind
            && let ExprKind::Lit(lit_right) = &right.kind
            && matches!(lit_right.node, LitKind::Int(..))
            && matches!(lit_left.node, LitKind::Int(..))
            && let ctxt = expr.span.ctxt()
            && ctxt == left.span.ctxt()
            && ctxt == right.span.ctxt()
            && ctxt == op.span.ctxt()
            && !expr.span.in_external_macro(cx.tcx.sess.source_map())
            && let Some(lit_right_src) = lit_right.span.get_text(cx)
            && lit_right_src
                .trim_start_matches(|c| is_whitespace(c) || c == '(')
                .strip_prefix('0')
                .and_then(|src| src.as_bytes().first().copied())
                .is_none_or(|c| !matches!(c, b'x' | b'X' | b'o' | b'O' | b'b' | b'B'))
        {
            span_lint_and_then(
                cx,
                SUSPICIOUS_XOR_USED_AS_POW,
                expr.span,
                "`^` is not the exponentiation operator",
                |diag| {
                    diag.span_suggestion_verbose(
                        expr.span,
                        "did you mean to write",
                        format!("{}.pow({})", lit_left.node, lit_right.node),
                        Applicability::MaybeIncorrect,
                    );
                },
            );
        }
    }
}
