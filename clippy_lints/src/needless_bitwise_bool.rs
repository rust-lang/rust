use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_opt;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{BinOpKind, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for uses of bitwise and/or operators between booleans, where performance may be improved by using
    /// a lazy and.
    ///
    /// ### Why is this bad?
    /// The bitwise operators do not support short-circuiting, so it may hinder code performance.
    /// Additionally, boolean logic "masked" as bitwise logic is not caught by lints like `unnecessary_fold`
    ///
    /// ### Known problems
    /// This lint evaluates only when the right side is determined to have no side effects. At this time, that
    /// determination is quite conservative.
    ///
    /// ### Example
    /// ```rust
    /// let (x,y) = (true, false);
    /// if x & !y {} // where both x and y are booleans
    /// ```
    /// Use instead:
    /// ```rust
    /// let (x,y) = (true, false);
    /// if x && !y {}
    /// ```
    #[clippy::version = "1.54.0"]
    pub NEEDLESS_BITWISE_BOOL,
    pedantic,
    "Boolean expressions that use bitwise rather than lazy operators"
}

declare_lint_pass!(NeedlessBitwiseBool => [NEEDLESS_BITWISE_BOOL]);

fn is_bitwise_operation(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    let ty = cx.typeck_results().expr_ty(expr);
    if_chain! {
        if !expr.span.from_expansion();
        if let (&ExprKind::Binary(ref op, _, right), &ty::Bool) = (&expr.kind, &ty.kind());
        if op.node == BinOpKind::BitAnd || op.node == BinOpKind::BitOr;
        if let ExprKind::Call(..) | ExprKind::MethodCall(..) | ExprKind::Binary(..) | ExprKind::Unary(..) = right.kind;
        if !right.can_have_side_effects();
        then {
            return true;
        }
    }
    false
}

fn suggession_snippet(cx: &LateContext<'_>, expr: &Expr<'_>) -> Option<String> {
    if let ExprKind::Binary(ref op, left, right) = expr.kind {
        if let (Some(l_snippet), Some(r_snippet)) = (snippet_opt(cx, left.span), snippet_opt(cx, right.span)) {
            let op_snippet = match op.node {
                BinOpKind::BitAnd => "&&",
                _ => "||",
            };
            return Some(format!("{} {} {}", l_snippet, op_snippet, r_snippet));
        }
    }
    None
}

impl LateLintPass<'_> for NeedlessBitwiseBool {
    fn check_expr(&mut self, cx: &LateContext<'_>, expr: &Expr<'_>) {
        if is_bitwise_operation(cx, expr) {
            span_lint_and_then(
                cx,
                NEEDLESS_BITWISE_BOOL,
                expr.span,
                "use of bitwise operator instead of lazy operator between booleans",
                |diag| {
                    if let Some(sugg) = suggession_snippet(cx, expr) {
                        diag.span_suggestion(expr.span, "try", sugg, Applicability::MachineApplicable);
                    }
                },
            );
        }
    }
}
