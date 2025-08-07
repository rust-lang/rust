use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::macros::{is_panic, root_macro_call};
use clippy_utils::{higher, is_else_clause, is_parent_stmt, peel_blocks_with_stmt, span_extract_comment, sugg};
use rustc_errors::Applicability;
use rustc_hir::{Expr, ExprKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::declare_lint_pass;

declare_clippy_lint! {
    /// ### What it does
    /// Detects `if`-then-`panic!` that can be replaced with `assert!`.
    ///
    /// ### Why is this bad?
    /// `assert!` is simpler than `if`-then-`panic!`.
    ///
    /// ### Example
    /// ```no_run
    /// let sad_people: Vec<&str> = vec![];
    /// if !sad_people.is_empty() {
    ///     panic!("there are sad people: {:?}", sad_people);
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
    /// let sad_people: Vec<&str> = vec![];
    /// assert!(sad_people.is_empty(), "there are sad people: {:?}", sad_people);
    /// ```
    #[clippy::version = "1.57.0"]
    pub MANUAL_ASSERT,
    pedantic,
    "`panic!` and only a `panic!` in `if`-then statement"
}

declare_lint_pass!(ManualAssert => [MANUAL_ASSERT]);

impl<'tcx> LateLintPass<'tcx> for ManualAssert {
    fn check_expr(&mut self, cx: &LateContext<'tcx>, expr: &Expr<'tcx>) {
        if let Some(higher::If { cond, then, r#else: None }) = higher::If::hir(expr)
            && !matches!(cond.kind, ExprKind::Let(_))
            && !expr.span.from_expansion()
            && let then = peel_blocks_with_stmt(then)
            && let Some(macro_call) = root_macro_call(then.span)
            && is_panic(cx, macro_call.def_id)
            && !cx.tcx.sess.source_map().is_multiline(cond.span)
            && let Ok(panic_snippet) = cx.sess().source_map().span_to_snippet(macro_call.span)
            && let Some(panic_snippet) = panic_snippet.strip_suffix(')')
            && let Some((_, format_args_snip)) = panic_snippet.split_once('(')
            // Don't change `else if foo { panic!(..) }` to `else { assert!(foo, ..) }` as it just
            // shuffles the condition around.
            // Should this have a config value?
            && !is_else_clause(cx.tcx, expr)
        {
            let mut applicability = Applicability::MachineApplicable;
            let mut comments = span_extract_comment(cx.sess().source_map(), expr.span);
            if !comments.is_empty() {
                comments += "\n";
            }
            let cond_sugg = !sugg::Sugg::hir_with_context(cx, cond, expr.span.ctxt(), "..", &mut applicability);
            let semicolon = if is_parent_stmt(cx, expr.hir_id) { ";" } else { "" };
            let sugg = format!("assert!({cond_sugg}, {format_args_snip}){semicolon}");
            // we show to the user the suggestion without the comments, but when applying the fix, include the
            // comments in the block
            span_lint_and_then(
                cx,
                MANUAL_ASSERT,
                expr.span,
                "only a `panic!` in `if`-then statement",
                |diag| {
                    // comments can be noisy, do not show them to the user
                    if !comments.is_empty() {
                        diag.tool_only_span_suggestion(
                            expr.span.shrink_to_lo(),
                            "add comments back",
                            comments,
                            applicability,
                        );
                    }
                    diag.span_suggestion(expr.span, "try instead", sugg, applicability);
                },
            );
        }
    }
}
