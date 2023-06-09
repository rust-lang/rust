use clippy_utils::{diagnostics::span_lint, source::snippet};
use itertools::Itertools;
use rustc_data_structures::fx::FxHashSet;
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::symbol::Ident;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for idents which comprise of a single letter.
    ///
    /// Note: This lint can be very noisy when enabled; it even lints generics! it may be desirable
    /// to only enable it temporarily.
    ///
    /// ### Why is this bad?
    /// In many cases it's not, but at times it can severely hinder readability. Some codebases may
    /// wish to disallow this to improve readability.
    ///
    /// ### Example
    /// ```rust,ignore
    /// for m in movies {
    ///	    let title = m.t;
    /// }
    /// ```
    /// Use instead:
    /// ```rust,ignore
    /// for movie in movies {
    ///     let title = movie.title;
    /// }
    /// ```
    /// ```
    #[clippy::version = "1.72.0"]
    pub SINGLE_CHAR_IDENTS,
    restriction,
    "disallows idents that can be represented as a char"
}
impl_lint_pass!(SingleCharIdents => [SINGLE_CHAR_IDENTS]);

#[derive(Clone)]
pub struct SingleCharIdents {
    pub allowed_idents: FxHashSet<char>,
}

impl EarlyLintPass for SingleCharIdents {
    fn check_ident(&mut self, cx: &EarlyContext<'_>, ident: Ident) {
        let str = ident.name.as_str();
        let chars = str.chars();
        if let [char, rest @ ..] = &*chars.collect_vec()
            && rest.is_empty()
            && self.allowed_idents.get(char).is_none()
            && !in_external_macro(cx.sess(), ident.span)
            // Ignore proc macros. Let's implement `WithSearchPat` for early lints someday :)
            && snippet(cx, ident.span, str) == str
        {
            span_lint(cx, SINGLE_CHAR_IDENTS, ident.span, "this ident comprises of a single char");
        }
    }
}
