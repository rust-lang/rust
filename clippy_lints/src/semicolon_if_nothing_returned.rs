use clippy_utils::diagnostics::span_lint_and_sugg;
use clippy_utils::source::snippet_with_context;
use rustc_errors::Applicability;
use rustc_hir::{Block, ExprKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::declare_lint_pass;
use rustc_span::{ExpnKind, MacroKind, Span};

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
    /// ```no_run
    /// fn main() {
    ///     println!("Hello world")
    /// }
    /// ```
    /// Use instead:
    /// ```no_run
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
        if !block.span.from_expansion()
            && let Some(expr) = block.expr
            && !from_attr_macro(expr.span)
            && let t_expr = cx.typeck_results().expr_ty(expr)
            && t_expr.is_unit()
            && let mut app = Applicability::MachineApplicable
            && let snippet = snippet_with_context(cx, expr.span, block.span.ctxt(), "}", &mut app).0
            && !snippet.ends_with('}')
            && !snippet.ends_with(';')
            && cx.sess().source_map().is_multiline(block.span)
        {
            // filter out the desugared `for` loop
            if let ExprKind::DropTemps(..) = &expr.kind {
                return;
            }
            span_lint_and_sugg(
                cx,
                SEMICOLON_IF_NOTHING_RETURNED,
                expr.span.source_callsite(),
                "consider adding a `;` to the last statement for consistent formatting",
                "add a `;` here",
                format!("{snippet};"),
                app,
            );
        }
    }
}

fn from_attr_macro(span: Span) -> bool {
    matches!(span.ctxt().outer_expn_data().kind, ExpnKind::Macro(MacroKind::Attr, _))
}
