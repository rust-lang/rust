use clippy_utils::diagnostics::{span_lint_and_help, span_lint_and_sugg};
use clippy_utils::source::snippet_opt;
use rustc_ast::ast;
use rustc_ast::tokenstream::TokenStream;
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::source_map::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of dbg!() macro.
    ///
    /// ### Why is this bad?
    /// `dbg!` macro is intended as a debugging tool. It
    /// should not be in version control.
    ///
    /// ### Known problems
    /// * The lint level is unaffected by crate attributes. The level can still
    ///   be set for functions, modules and other items. To change the level for
    ///   the entire crate, please use command line flags. More information and a
    ///   configuration example can be found in [clippy#6610].
    ///
    /// [clippy#6610]: https://github.com/rust-lang/rust-clippy/issues/6610#issuecomment-977120558
    ///
    /// ### Example
    /// ```rust,ignore
    /// // Bad
    /// dbg!(true)
    ///
    /// // Good
    /// true
    /// ```
    #[clippy::version = "1.34.0"]
    pub DBG_MACRO,
    restriction,
    "`dbg!` macro is intended as a debugging tool"
}

declare_lint_pass!(DbgMacro => [DBG_MACRO]);

impl EarlyLintPass for DbgMacro {
    fn check_mac(&mut self, cx: &EarlyContext<'_>, mac: &ast::MacCall) {
        if mac.path == sym!(dbg) {
            if let Some(sugg) = tts_span(mac.args.inner_tokens()).and_then(|span| snippet_opt(cx, span)) {
                span_lint_and_sugg(
                    cx,
                    DBG_MACRO,
                    mac.span(),
                    "`dbg!` macro is intended as a debugging tool",
                    "ensure to avoid having uses of it in version control",
                    sugg,
                    Applicability::MaybeIncorrect,
                );
            } else {
                span_lint_and_help(
                    cx,
                    DBG_MACRO,
                    mac.span(),
                    "`dbg!` macro is intended as a debugging tool",
                    None,
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
    let span = cursor.last().map_or(first, |tree| first.to(tree.span()));
    Some(span)
}
