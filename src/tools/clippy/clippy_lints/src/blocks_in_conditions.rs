use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_block_with_applicability;
use clippy_utils::{contains_return, higher, is_from_proc_macro};
use rustc_errors::Applicability;
use rustc_hir::{BlockCheckMode, Expr, ExprKind, MatchSource};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `if` and `match` conditions that use blocks containing an
    /// expression, statements or conditions that use closures with blocks.
    ///
    /// ### Why is this bad?
    /// Style, using blocks in the condition makes it hard to read.
    ///
    /// ### Examples
    /// ```no_run
    /// # fn somefunc() -> bool { true };
    /// if { true } { /* ... */ }
    ///
    /// if { let x = somefunc(); x } { /* ... */ }
    ///
    /// match { let e = somefunc(); e } {
    ///     // ...
    /// #   _ => {}
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// # fn somefunc() -> bool { true };
    /// if true { /* ... */ }
    ///
    /// let res = { let x = somefunc(); x };
    /// if res { /* ... */ }
    ///
    /// let res = { let e = somefunc(); e };
    /// match res {
    ///     // ...
    /// #   _ => {}
    /// }
    /// ```
    #[clippy::version = "1.45.0"]
    pub BLOCKS_IN_CONDITIONS,
    style,
    "useless or complex blocks that can be eliminated in conditions"
}

declare_lint_pass!(BlocksInConditions => [BLOCKS_IN_CONDITIONS]);

const BRACED_EXPR_MESSAGE: &str = "omit braces around single expression condition";

impl<'tcx> LateLintPass<'tcx> for BlocksInConditions {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if expr.span.in_external_macro(cx.sess().source_map()) {
            return;
        }

        let Some((cond, keyword, desc)) = higher::If::hir(expr)
            .map(|hif| (hif.cond, "if", "an `if` condition"))
            .or(if let ExprKind::Match(match_ex, _, MatchSource::Normal) = expr.kind {
                Some((match_ex, "match", "a `match` scrutinee"))
            } else {
                None
            })
        else {
            return;
        };
        let complex_block_message = format!(
            "in {desc}, avoid complex blocks or closures with blocks; \
            instead, move the block or closure higher and bind it with a `let`",
        );

        if let ExprKind::Block(block, _) = &cond.kind {
            if !block.span.eq_ctxt(expr.span) {
                // If the block comes from a macro, or as an argument to a macro,
                // do not lint.
                return;
            }
            if block.rules == BlockCheckMode::DefaultBlock {
                if block.stmts.is_empty() {
                    if let Some(ex) = &block.expr {
                        // don't dig into the expression here, just suggest that they remove
                        // the block
                        if expr.span.from_expansion() || ex.span.from_expansion() {
                            return;
                        }

                        // Linting should not be triggered to cases where `return` is included in the condition.
                        // #9911
                        if contains_return(block.expr) {
                            return;
                        }

                        let mut applicability = Applicability::MachineApplicable;
                        span_lint_and_sugg(
                            cx,
                            BLOCKS_IN_CONDITIONS,
                            cond.span,
                            BRACED_EXPR_MESSAGE,
                            "try",
                            snippet_block_with_applicability(cx, ex.span, "..", Some(expr.span), &mut applicability),
                            applicability,
                        );
                    }
                } else {
                    let span = block.expr.as_ref().map_or_else(|| block.stmts[0].span, |e| e.span);
                    if span.from_expansion() || expr.span.from_expansion() || is_from_proc_macro(cx, cond) {
                        return;
                    }
                    // move block higher
                    let mut applicability = Applicability::MachineApplicable;
                    span_lint_and_sugg(
                        cx,
                        BLOCKS_IN_CONDITIONS,
                        expr.span.with_hi(cond.span.hi()),
                        complex_block_message,
                        "try",
                        format!(
                            "let res = {}; {keyword} res",
                            snippet_block_with_applicability(cx, block.span, "..", Some(expr.span), &mut applicability),
                        ),
                        applicability,
                    );
                }
            }
        }
    }
}
