use clippy_utils::diagnostics::span_lint;
use itertools::Itertools;
use rustc_ast::{AttrKind, Attribute};
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

declare_clippy_lint! {
    /// ### What it does
    /// Detects the syntax `['foo']` in documentation comments (notice quotes instead of backticks)
    /// outside of code blocks
    /// ### Why is this bad?
    /// It is likely a typo when defining an intra-doc link
    ///
    /// ### Example
    /// ```rust
    /// /// See also: ['foo']
    /// fn bar() {}
    /// ```
    /// Use instead:
    /// ```rust
    /// /// See also: [`foo`]
    /// fn bar() {}
    /// ```
    #[clippy::version = "1.63.0"]
    pub DOC_LINK_WITH_QUOTES,
    pedantic,
    "possible typo for an intra-doc link"
}
declare_lint_pass!(DocLinkWithQuotes => [DOC_LINK_WITH_QUOTES]);

impl EarlyLintPass for DocLinkWithQuotes {
    fn check_attribute(&mut self, ctx: &EarlyContext<'_>, attr: &Attribute) {
        if let AttrKind::DocComment(_, symbol) = attr.kind {
            if contains_quote_link(symbol.as_str()) {
                span_lint(
                    ctx,
                    DOC_LINK_WITH_QUOTES,
                    attr.span,
                    "possible intra-doc link using quotes instead of backticks",
                );
            }
        }
    }
}

fn contains_quote_link(s: &str) -> bool {
    let mut in_backticks = false;
    let mut found_opening = false;

    for c in s.chars().tuple_windows::<(char, char)>() {
        match c {
            ('`', _) => in_backticks = !in_backticks,
            ('[', '\'') if !in_backticks => found_opening = true,
            ('\'', ']') if !in_backticks && found_opening => return true,
            _ => {},
        }
    }

    false
}
