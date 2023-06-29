use clippy_utils::{diagnostics::span_lint_and_sugg, higher::If, is_from_proc_macro, source::snippet_opt};
use rustc_errors::Applicability;
use rustc_hir::{ExprKind, Stmt, StmtKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Checks for empty `if` branches with no else branch.
    ///
    /// ### Why is this bad?
    /// It can be entirely omitted, and often the condition too.
    ///
    /// ### Known issues
    /// This will usually only suggest to remove the `if` statement, not the condition. Other lints
    /// such as `no_effect` will take care of removing the condition if it's unnecessary.
    ///
    /// ### Example
    /// ```rust,ignore
    /// if really_expensive_condition(&i) {}
    /// if really_expensive_condition_with_side_effects(&mut i) {}
    /// ```
    /// Use instead:
    /// ```rust,ignore
    /// // <omitted>
    /// really_expensive_condition_with_side_effects(&mut i);
    /// ```
    #[clippy::version = "1.72.0"]
    pub NEEDLESS_IF,
    complexity,
    "checks for empty if branches"
}
declare_lint_pass!(NeedlessIf => [NEEDLESS_IF]);

impl LateLintPass<'_> for NeedlessIf {
    fn check_stmt<'tcx>(&mut self, cx: &LateContext<'tcx>, stmt: &Stmt<'tcx>) {
        if let StmtKind::Expr(expr) = stmt.kind
            && let Some(If {cond, then, r#else: None }) = If::hir(expr)
            && let ExprKind::Block(block, ..) = then.kind
            && block.stmts.is_empty()
            && block.expr.is_none()
            && !in_external_macro(cx.sess(), expr.span)
            && !is_from_proc_macro(cx, expr)
            && let Some(then_snippet) = snippet_opt(cx, then.span)
            // Ignore
            // - empty macro expansions
            // - empty reptitions in macro expansions
            // - comments
            // - #[cfg]'d out code
            && then_snippet.chars().all(|ch| matches!(ch, '{' | '}') || ch.is_ascii_whitespace())
            && let Some(cond_snippet) = snippet_opt(cx, cond.span)
        {
            span_lint_and_sugg(
                cx,
                NEEDLESS_IF,
                stmt.span,
                "this `if` branch is empty",
                "you can remove it",
                if cond.can_have_side_effects() || !cx.tcx.hir().attrs(stmt.hir_id).is_empty() {
                    // `{ foo }` or `{ foo } && bar` placed into a statement position would be
                    // interpreted as a block statement, force it to be an expression
                    if cond_snippet.starts_with('{') {
                        format!("({cond_snippet});")
                    } else {
                        format!("{cond_snippet};")
                    }
                } else {
                    String::new()
                },
                Applicability::MachineApplicable,
            );
        }
    }
}
