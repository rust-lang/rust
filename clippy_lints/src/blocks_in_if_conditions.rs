use clippy_utils::diagnostics::{span_lint, span_lint_and_sugg};
use clippy_utils::get_parent_expr;
use clippy_utils::higher;
use clippy_utils::source::snippet_block_with_applicability;
use clippy_utils::ty::implements_trait;
use clippy_utils::visitors::{for_each_expr, Descend};
use core::ops::ControlFlow;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{BlockCheckMode, Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for `if` conditions that use blocks containing an
    /// expression, statements or conditions that use closures with blocks.
    ///
    /// ### Why is this bad?
    /// Style, using blocks in the condition makes it hard to read.
    ///
    /// ### Examples
    /// ```rust
    /// # fn somefunc() -> bool { true };
    /// if { true } { /* ... */ }
    ///
    /// if { let x = somefunc(); x } { /* ... */ }
    /// ```
    ///
    /// Use instead:
    /// ```rust
    /// # fn somefunc() -> bool { true };
    /// if true { /* ... */ }
    ///
    /// let res = { let x = somefunc(); x };
    /// if res { /* ... */ }
    /// ```
    #[clippy::version = "1.45.0"]
    pub BLOCKS_IN_IF_CONDITIONS,
    style,
    "useless or complex blocks that can be eliminated in conditions"
}

declare_lint_pass!(BlocksInIfConditions => [BLOCKS_IN_IF_CONDITIONS]);

const BRACED_EXPR_MESSAGE: &str = "omit braces around single expression condition";
const COMPLEX_BLOCK_MESSAGE: &str = "in an `if` condition, avoid complex blocks or closures with blocks; \
                                    instead, move the block or closure higher and bind it with a `let`";

impl<'tcx> LateLintPass<'tcx> for BlocksInIfConditions {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &'tcx Expr<'_>) {
        if in_external_macro(cx.sess(), expr.span) {
            return;
        }
        if let Some(higher::If { cond, .. }) = higher::If::hir(expr) {
            if let ExprKind::Block(block, _) = &cond.kind {
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
                                BLOCKS_IN_IF_CONDITIONS,
                                cond.span,
                                BRACED_EXPR_MESSAGE,
                                "try",
                                format!(
                                    "{}",
                                    snippet_block_with_applicability(
                                        cx,
                                        ex.span,
                                        "..",
                                        Some(expr.span),
                                        &mut applicability
                                    )
                                ),
                                applicability,
                            );
                        }
                    } else {
                        let span = block.expr.as_ref().map_or_else(|| block.stmts[0].span, |e| e.span);
                        if span.from_expansion() || expr.span.from_expansion() {
                            return;
                        }
                        // move block higher
                        let mut applicability = Applicability::MachineApplicable;
                        span_lint_and_sugg(
                            cx,
                            BLOCKS_IN_IF_CONDITIONS,
                            expr.span.with_hi(cond.span.hi()),
                            COMPLEX_BLOCK_MESSAGE,
                            "try",
                            format!(
                                "let res = {}; if res",
                                snippet_block_with_applicability(
                                    cx,
                                    block.span,
                                    "..",
                                    Some(expr.span),
                                    &mut applicability
                                ),
                            ),
                            applicability,
                        );
                    }
                }
            } else {
                let _: Option<!> = for_each_expr(cond, |e| {
                    if let ExprKind::Closure(closure) = e.kind {
                        // do not lint if the closure is called using an iterator (see #1141)
                        if_chain! {
                            if let Some(parent) = get_parent_expr(cx, e);
                            if let ExprKind::MethodCall(_, self_arg, _, _) = &parent.kind;
                            let caller = cx.typeck_results().expr_ty(self_arg);
                            if let Some(iter_id) = cx.tcx.get_diagnostic_item(sym::Iterator);
                            if implements_trait(cx, caller, iter_id, &[]);
                            then {
                                return ControlFlow::Continue(Descend::No);
                            }
                        }

                        let body = cx.tcx.hir().body(closure.body);
                        let ex = &body.value;
                        if let ExprKind::Block(block, _) = ex.kind {
                            if !body.value.span.from_expansion() && !block.stmts.is_empty() {
                                span_lint(cx, BLOCKS_IN_IF_CONDITIONS, ex.span, COMPLEX_BLOCK_MESSAGE);
                                return ControlFlow::Continue(Descend::No);
                            }
                        }
                    }
                    ControlFlow::Continue(Descend::Yes)
                });
            }
        }
    }
}
