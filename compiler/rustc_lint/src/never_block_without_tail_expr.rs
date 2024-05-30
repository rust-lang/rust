use crate::{LateContext, LateLintPass, LintContext};
use rustc_session::lint::FutureIncompatibilityReason;

use rustc_hir::{Block, Expr, ExprKind, StmtKind};
use rustc_macros::{LintDiagnostic, Subdiagnostic};
use rustc_session::{declare_lint, declare_lint_pass};
use rustc_span::{edition::Edition, sym, Span, SyntaxContext};
use tracing::debug;

declare_lint! {
    /// The `never_block_without_tail_expr` lint checks for blocks which have type `!`, but do not
    /// end in an expression or a `return`, `break`, or `continue` statement
    ///
    /// ### Example
    ///
    /// ```rust
    /// let _: u8 = {
    ///     return 1;
    /// };
    /// ```
    ///
    /// {{produces}}
    ///
    /// ### Explanation
    ///
    /// Previously, blocks which diverged in all branches (e.g. by having a `return`) had type `!`
    /// (which can be coerced to any type). This is being changed in the 2024 edition.
    pub NEVER_BLOCK_WITHOUT_TAIL_EXPR,
    Warn,
    "`Iterator::map` call that discard the iterator's values",
    @future_incompatible = FutureIncompatibleInfo {
        reason: FutureIncompatibilityReason::EditionError(Edition::Edition2024),
        reference: "https://github.com/rust-lang/rust/issues/123747",
    };
}

declare_lint_pass!(NeverBlockWithoutTailExpr => [NEVER_BLOCK_WITHOUT_TAIL_EXPR]);

impl<'tcx> LateLintPass<'tcx> for NeverBlockWithoutTailExpr {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &Expr<'_>) {
        // A block without tail expression
        let ExprKind::Block(Block { stmts, expr: None, .. }, ..) = expr.kind else { return };

        // Last statement of which is not `return`, `break`, `continue`, or `become`
        if let [.., last] = stmts
            && let StmtKind::Semi(e) = last.kind
            && matches!(e.kind, ExprKind::Ret(_) | ExprKind::Break(_, _) | ExprKind::Continue(_))
        {
            return;
        }

        // Which has type `!`, which is coerced to non-unit
        let typeck = cx.typeck_results();
        debug!(block_ty = ?typeck.expr_ty(expr), block_ty_adjusted = ?typeck.expr_ty_adjusted(expr));
        if !typeck.expr_ty(expr).is_never() || typeck.expr_ty_adjusted(expr).is_unit() {
            return;
        }

        // If last statement has type `!` suggest turning it into a tail expression,
        // otherwise provide a generic help message suggesting possible solutions.
        let help = if let [.., last] = stmts
            && let StmtKind::Semi(e) = last.kind
            // Due to how never type fallback works, the panic macro does not have type `!`,
            // fallback happens before this lint (which on edition < 2024 sets the type to `()`),
            // and the adjustment is on some inner level inside macro.
            //
            // Special casing std/core panic-like macros is not nice, but I do not see a better solution.
            && (typeck.expr_ty(e).is_never() || is_panic_macro_expansion(cx, e))
            // Climb out of macros, so that we get span of the call,
            // instead of the span of the macro internals
            && let e_span =
                e.span.find_oldest_ancestor_in_same_ctxt().with_ctxt(SyntaxContext::root())
            // Get the span between the expression and the end of the statement,
            // i.e. the span of the semicolon
            && let span = e_span.between(last.span.shrink_to_hi())
            // TODO: remove this workaround for macros misbehaving
            //       (`;` is not included in the span of the statement after macro expansion,
            //        so the span ends up being empty, so debug assertions in diagnostic code fail)
            && !span.is_empty()
        {
            Help::RemoveSemi { span }
        } else {
            Help::Generic
        };

        cx.emit_span_lint(
            NEVER_BLOCK_WITHOUT_TAIL_EXPR,
            expr.span,
            NeverBlockWithoutTailExprLint { help },
        );
    }
}

fn is_panic_macro_expansion(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    let Some(id) = expr.span.ctxt().outer_expn_data().macro_def_id else { return false };
    let Some(name) = cx.tcx.get_diagnostic_name(id) else { return false };

    matches!(
        name,
        sym::core_panic_macro
            | sym::core_panic_2015_macro
            | sym::core_panic_2021_macro
            | sym::std_panic_macro
            | sym::std_panic_2015_macro
            | sym::unreachable_macro
            | sym::todo_macro
    )
}

#[derive(LintDiagnostic)]
#[diag(lint_never_block_without_tail_expr)]
#[note]
struct NeverBlockWithoutTailExprLint {
    #[subdiagnostic]
    help: Help,
}

#[derive(Subdiagnostic)]
enum Help {
    #[suggestion(
        lint_never_block_without_tail_expr_help_remove_semi,
        style = "short",
        code = "",
        applicability = "machine-applicable"
    )]
    RemoveSemi {
        #[primary_span]
        span: Span,
    },

    #[help(lint_never_block_without_tail_expr_help_generic)]
    Generic,
}
