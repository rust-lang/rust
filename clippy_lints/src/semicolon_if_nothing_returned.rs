use crate::utils::{in_macro, span_lint_and_sugg, sugg};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{Block, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:** Looks for blocks of expressions and fires if the last expression returns `()`
    /// but is not followed by a semicolon.
    ///
    /// **Why is this bad?** The semicolon might be optional but when
    /// extending the block with new code, it doesn't require a change in previous last line.
    /// It's also more idiomatic.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// fn main() {
    ///     println!("Hello world")
    /// }
    /// ```
    /// Use instead:
    /// ```rust
    /// fn main() {
    ///     println!("Hello world");
    /// }
    /// ```
    pub SEMICOLON_IF_NOTHING_RETURNED,
    pedantic,
    "add a semicolon if nothing is returned"
}

declare_lint_pass!(SemicolonIfNothingReturned => [SEMICOLON_IF_NOTHING_RETURNED]);

impl LateLintPass<'_> for SemicolonIfNothingReturned {
    fn check_block(&mut self, cx: &LateContext<'tcx>, block: &'tcx Block<'tcx>) {
        if_chain! {
            if !in_macro(block.span);
            if let Some(expr) = block.expr;
            let t_expr = cx.typeck_results().expr_ty(expr);
            if t_expr.is_unit();
            if let Ok(snippet) = cx.tcx.sess.source_map().span_to_snippet(expr.span.source_callsite());
            if !snippet.ends_with('}');
            then {
                // filter out the desugared `for` loop
                if let ExprKind::DropTemps(..) = &expr.kind {
                    return;
                }

                let sugg = sugg::Sugg::hir(cx, &expr, "..");
                let suggestion = format!("{0};", sugg);
                span_lint_and_sugg(
                    cx,
                    SEMICOLON_IF_NOTHING_RETURNED,
                    expr.span,
                    "Consider adding a `;` to the last statement for consistent formatting",
                    "add a `;` here",
                    suggestion,
                    Applicability::MaybeIncorrect,
                );
            }
        }
    }
}
