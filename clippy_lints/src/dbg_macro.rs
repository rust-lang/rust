use crate::utils::{snippet_opt, span_help_and_lint, span_lint_and_sugg};
use rustc::lint::{EarlyContext, EarlyLintPass, LintArray, LintPass};
use rustc::{declare_tool_lint, lint_array};
use rustc_errors::Applicability;
use syntax::ast;
use syntax::source_map::Span;
use syntax::tokenstream::TokenStream;

declare_clippy_lint! {
    /// **What it does:** Checks for usage of dbg!() macro.
    ///
    /// **Why is this bad?** `dbg!` macro is intended as a debugging tool. It
    /// should not be in version control.
    ///
    /// **Known problems:** None.
    ///
    /// **Example:**
    /// ```rust,ignore
    /// // Bad
    /// dbg!(true)
    ///
    /// // Good
    /// true
    /// ```
    pub DBG_MACRO,
    restriction,
    "`dbg!` macro is intended as a debugging tool"
}

#[derive(Copy, Clone, Debug)]
pub struct Pass;

impl LintPass for Pass {
    fn get_lints(&self) -> LintArray {
        lint_array!(DBG_MACRO)
    }

    fn name(&self) -> &'static str {
        "DbgMacro"
    }
}

impl EarlyLintPass for Pass {
    fn check_mac(&mut self, cx: &EarlyContext<'_>, mac: &ast::Mac) {
        if mac.node.path == "dbg" {
            if let Some(sugg) = tts_span(mac.node.tts.clone()).and_then(|span| snippet_opt(cx, span)) {
                span_lint_and_sugg(
                    cx,
                    DBG_MACRO,
                    mac.span,
                    "`dbg!` macro is intended as a debugging tool",
                    "ensure to avoid having uses of it in version control",
                    sugg,
                    Applicability::MaybeIncorrect,
                );
            } else {
                span_help_and_lint(
                    cx,
                    DBG_MACRO,
                    mac.span,
                    "`dbg!` macro is intended as a debugging tool",
                    "ensure to avoid having uses of it in version control",
                );
            }
        }
    }
}

// Get span enclosing entire the token stream.
fn tts_span(tts: TokenStream) -> Option<Span> {
    let mut cursor = tts.into_trees();
    let first = cursor.next()?.span();
    let span = match cursor.last() {
        Some(tree) => first.to(tree.span()),
        None => first,
    };
    Some(span)
}
