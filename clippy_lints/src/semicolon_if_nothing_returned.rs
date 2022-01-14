use crate::rustc_lint::LintContext;
use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_macro_callsite;
use clippy_utils::sugg;
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::{Block, ExprKind};
use rustc_lint::{LateContext, LateLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Looks for blocks of expressions and fires if the last expression returns
    /// `()` but is not followed by a semicolon.
    ///
    /// ### Why is this bad?
    /// The semicolon might be optional but when extending the block with new
    /// code, it doesn't require a change in previous last line.
    ///
    /// ### Example
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
    #[clippy::version = "1.52.0"]
    pub SEMICOLON_IF_NOTHING_RETURNED,
    pedantic,
    "add a semicolon if nothing is returned"
}

declare_lint_pass!(SemicolonIfNothingReturned => [SEMICOLON_IF_NOTHING_RETURNED]);

impl<'tcx> LateLintPass<'tcx> for SemicolonIfNothingReturned {
    fn check_block(&mut self, cx: &LateContext<'tcx>, block: &'tcx Block<'tcx>) {
        if_chain! {
            if !block.span.from_expansion();
            if let Some(expr) = block.expr;
            let t_expr = cx.typeck_results().expr_ty(expr);
            if t_expr.is_unit();
            if let snippet = snippet_with_macro_callsite(cx, expr.span, "}");
            if !snippet.ends_with('}') && !snippet.ends_with(';');
            if cx.sess().source_map().is_multiline(block.span);
            then {
                // filter out the desugared `for` loop
                if let ExprKind::DropTemps(..) = &expr.kind {
                    return;
                }

                let sugg = sugg::Sugg::hir_with_macro_callsite(cx, expr, "..");
                let suggestion = format!("{0};", sugg);
                span_lint_and_sugg(
                    cx,
                    SEMICOLON_IF_NOTHING_RETURNED,
                    expr.span.source_callsite(),
                    "consider adding a `;` to the last statement for consistent formatting",
                    "add a `;` here",
                    suggestion,
                    Applicability::MaybeIncorrect,
                );
            }
        }
    }
}
