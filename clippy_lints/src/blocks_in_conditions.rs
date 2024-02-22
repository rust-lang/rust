use clippy_utils::diagnostics::{span_lint, span_lint_and_sugg};
use clippy_utils::source::snippet_block_with_applicability;
use clippy_utils::ty::implements_trait;
use clippy_utils::visitors::{for_each_expr, Descend};
use clippy_utils::{get_parent_expr, higher, is_from_proc_macro};
use core::ops::ControlFlow;
use rustc_errors::Applicability;
use rustc_hir::{BlockCheckMode, Expr, ExprKind, MatchSource};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::declare_lint_pass;
use rustc_span::sym;

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
        if in_external_macro(cx.sess(), expr.span) {
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
        let complex_block_message = &format!(
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
                        let mut applicability = Applicability::MachineApplicable;
                        span_lint_and_sugg(
                            cx,
                            BLOCKS_IN_CONDITIONS,
                            cond.span,
                            BRACED_EXPR_MESSAGE,
                            "try",
                            snippet_block_with_applicability(cx, ex.span, "..", Some(expr.span), &mut applicability)
                                .to_string(),
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
        } else {
            let _: Option<!> = for_each_expr(cond, |e| {
                if let ExprKind::Closure(closure) = e.kind {
                    // do not lint if the closure is called using an iterator (see #1141)
                    if let Some(parent) = get_parent_expr(cx, e)
                        && let ExprKind::MethodCall(_, self_arg, _, _) = &parent.kind
                        && let caller = cx.typeck_results().expr_ty(self_arg)
                        && let Some(iter_id) = cx.tcx.get_diagnostic_item(sym::Iterator)
                        && implements_trait(cx, caller, iter_id, &[])
                    {
                        return ControlFlow::Continue(Descend::No);
                    }

                    let body = cx.tcx.hir().body(closure.body);
                    let ex = &body.value;
                    if let ExprKind::Block(block, _) = ex.kind {
                        if !body.value.span.from_expansion() && !block.stmts.is_empty() {
                            span_lint(cx, BLOCKS_IN_CONDITIONS, ex.span, complex_block_message);
                            return ControlFlow::Continue(Descend::No);
                        }
                    }
                }
                ControlFlow::Continue(Descend::Yes)
            });
        }
    }
}
