use clippy_utils::consts::{ConstEvalCtxt, Constant};
use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::is_else_clause;
use rustc_hir::{BinOpKind, Expr, ExprKind, UnOp};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `!` or `!=` in an if condition with an
    /// else branch.
    ///
    /// ### Why is this bad?
    /// Negations reduce the readability of statements.
    ///
    /// ### Example
    /// ```no_run
    /// # let v: Vec<usize> = vec![];
    /// # fn a() {}
    /// # fn b() {}
    /// if !v.is_empty() {
    ///     a()
    /// } else {
    ///     b()
    /// }
    /// ```
    ///
    /// Could be written:
    ///
    /// ```no_run
    /// # let v: Vec<usize> = vec![];
    /// # fn a() {}
    /// # fn b() {}
    /// if v.is_empty() {
    ///     b()
    /// } else {
    ///     a()
    /// }
    /// ```
    #[clippy::version = "pre 1.29.0"]
    pub IF_NOT_ELSE,
    pedantic,
    "`if` branches that could be swapped so no negation operation is necessary on the condition"
}

declare_lint_pass!(IfNotElse => [IF_NOT_ELSE]);

fn is_zero_const(expr: &Expr<'_>, cx: &LateContext<'_>) -> bool {
    if let Some(value) = ConstEvalCtxt::new(cx).eval_simple(expr) {
        return Constant::Int(0) == value;
    }
    false
}

impl LateLintPass<'_> for IfNotElse {
    fn check_expr(&mut self, cx: &LateContext<'_>, e: &Expr<'_>) {
        if let ExprKind::If(cond, _, Some(els)) = e.kind
            && let ExprKind::DropTemps(cond) = cond.kind
            && let ExprKind::Block(..) = els.kind
        {
            let (msg, help) = match cond.kind {
                ExprKind::Unary(UnOp::Not, _) => (
                    "unnecessary boolean `not` operation",
                    "remove the `!` and swap the blocks of the `if`/`else`",
                ),
                // Don't lint on `… != 0`, as these are likely to be bit tests.
                // For example, `if foo & 0x0F00 != 0 { … } else { … }` is already in the "proper" order.
                ExprKind::Binary(op, _, rhs) if op.node == BinOpKind::Ne && !is_zero_const(rhs, cx) => (
                    "unnecessary `!=` operation",
                    "change to `==` and swap the blocks of the `if`/`else`",
                ),
                _ => return,
            };

            // `from_expansion` will also catch `while` loops which appear in the HIR as:
            // ```rust
            // loop {
            //     if cond { ... } else { break; }
            // }
            // ```
            if !e.span.from_expansion() && !is_else_clause(cx.tcx, e) {
                span_lint_and_help(cx, IF_NOT_ELSE, e.span, msg, None, help);
            }
        }
    }
}
