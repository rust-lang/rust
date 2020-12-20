use crate::utils::{match_def_path, paths, span_lint_and_then, sugg};
use if_chain::if_chain;
use rustc_errors::Applicability;
use rustc_hir::*;
use rustc_lint::{LateContext, LateLintPass};
use rustc_middle::ty;
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// **What it does:**
    ///
    /// **Why is this bad?**
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// // example code where clippy issues a warning
    /// ```
    /// Use instead:
    /// ```rust
    /// // example code which does not raise clippy warning
    /// ```
    pub SEMICOLON_IF_NOTHING_RETURNED,
    pedantic,
    "default lint description"
}

declare_lint_pass!(SemicolonIfNothingReturned => [SEMICOLON_IF_NOTHING_RETURNED]);

impl LateLintPass<'_> for SemicolonIfNothingReturned {
    fn check_block(&mut self, cx: &LateContext<'tcx>, block: &'tcx Block<'tcx>) {
        if_chain! {
            if let Some(expr) = block.expr;
            let t_expr = cx.typeck_results().expr_ty(expr);
            if t_expr.is_unit();
            then {
                let sugg = sugg::Sugg::hir(cx, &expr, "..");
                let suggestion = format!("{0};", sugg);
                span_lint_and_then(
                    cx,
                    SEMICOLON_IF_NOTHING_RETURNED,
                    expr.span,
                    "add `;` to terminate block",
                    | diag | {
                        diag.span_suggestion(
                            expr.span,
                            "add `;`",
                            suggestion,
                            Applicability::MaybeIncorrect,
                        );
                    }
                )
            }
        }
    }
}
