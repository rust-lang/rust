//! lint on if branches that could be swapped so no `!` operation is necessary
//! on the condition

use clippy_utils::consts::{constant_simple, Constant};
use clippy_utils::diagnostics::span_lint_and_help;
use clippy_utils::is_else_clause;
use rustc_hir::{BinOpKind, Expr, ExprKind, UnOp};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

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
    if let Some(value) = constant_simple(cx, cx.typeck_results(), expr) {
        return Constant::Int(0) == value;
    }
    false
}

impl LateLintPass<'_> for IfNotElse {
    fn check_expr(&mut self, cx: &LateContext<'_>, item: &Expr<'_>) {
        // While loops will be desugared to ExprKind::If. This will cause the lint to fire.
        // To fix this, return early if this span comes from a macro or desugaring.
        if item.span.from_expansion() {
            return;
        }
        if let ExprKind::If(cond, _, Some(els)) = item.kind {
            if let ExprKind::Block(..) = els.kind {
                // Disable firing the lint in "else if" expressions.
                if is_else_clause(cx.tcx, item) {
                    return;
                }

                match cond.peel_drop_temps().kind {
                    ExprKind::Unary(UnOp::Not, _) => {
                        span_lint_and_help(
                            cx,
                            IF_NOT_ELSE,
                            item.span,
                            "unnecessary boolean `not` operation",
                            None,
                            "remove the `!` and swap the blocks of the `if`/`else`",
                        );
                    },
                    ExprKind::Binary(ref kind, _, lhs) if kind.node == BinOpKind::Ne && !is_zero_const(lhs, cx) => {
                        // Disable firing the lint on `… != 0`, as these are likely to be bit tests.
                        // For example, `if foo & 0x0F00 != 0 { … } else { … }` already is in the "proper" order.
                        span_lint_and_help(
                            cx,
                            IF_NOT_ELSE,
                            item.span,
                            "unnecessary `!=` operation",
                            None,
                            "change to `==` and swap the blocks of the `if`/`else`",
                        );
                    },
                    _ => (),
                }
            }
        }
    }
}
