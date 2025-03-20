use clippy_utils::diagnostics::span_lint_and_sugg;
use rustc_ast::ast::{AttrKind, Attribute, Item, ItemKind};
use rustc_ast::token::{Token, TokenKind};
use rustc_ast::tokenstream::{TokenStream, TokenTree};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::Span;
use rustc_span::symbol::sym;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for usage of `crate` as opposed to `$crate` in a macro definition.
    ///
    /// ### Why is this bad?
    /// `crate` refers to the macro call's crate, whereas `$crate` refers to the macro definition's
    /// crate. Rarely is the former intended. See:
    /// https://doc.rust-lang.org/reference/macros-by-example.html#hygiene
    ///
    /// ### Example
    /// ```no_run
    /// #[macro_export]
    /// macro_rules! print_message {
    ///     () => {
    ///         println!("{}", crate::MESSAGE);
    ///     };
    /// }
    /// pub const MESSAGE: &str = "Hello!";
    /// ```
    /// Use instead:
    /// ```no_run
    /// #[macro_export]
    /// macro_rules! print_message {
    ///     () => {
    ///         println!("{}", $crate::MESSAGE);
    ///     };
    /// }
    /// pub const MESSAGE: &str = "Hello!";
    /// ```
    ///
    /// Note that if the use of `crate` is intentional, an `allow` attribute can be applied to the
    /// macro definition, e.g.:
    /// ```rust,ignore
    /// #[allow(clippy::crate_in_macro_def)]
    /// macro_rules! ok { ... crate::foo ... }
    /// ```
    #[clippy::version = "1.62.0"]
    pub CRATE_IN_MACRO_DEF,
    suspicious,
    "using `crate` in a macro definition"
}
declare_lint_pass!(CrateInMacroDef => [CRATE_IN_MACRO_DEF]);

impl EarlyLintPass for CrateInMacroDef {
    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &Item) {
        if let ItemKind::MacroDef(_, macro_def) = &item.kind
            && item.attrs.iter().any(is_macro_export)
            && let Some(span) = contains_unhygienic_crate_reference(&macro_def.body.tokens)
        {
            span_lint_and_sugg(
                cx,
                CRATE_IN_MACRO_DEF,
                span,
                "`crate` references the macro call's crate",
                "to reference the macro definition's crate, use",
                String::from("$crate"),
                Applicability::MachineApplicable,
            );
        }
    }
}

fn is_macro_export(attr: &Attribute) -> bool {
    if let AttrKind::Normal(normal) = &attr.kind
        && let [segment] = normal.item.path.segments.as_slice()
    {
        segment.ident.name == sym::macro_export
    } else {
        false
    }
}

fn contains_unhygienic_crate_reference(tts: &TokenStream) -> Option<Span> {
    let mut prev_is_dollar = false;
    let mut iter = tts.iter();
    while let Some(curr) = iter.next() {
        if !prev_is_dollar
            && let Some(span) = is_crate_keyword(curr)
            && let Some(next) = iter.peek()
            && is_token(next, &TokenKind::PathSep)
        {
            return Some(span);
        }
        if let TokenTree::Delimited(.., tts) = &curr {
            let span = contains_unhygienic_crate_reference(tts);
            if span.is_some() {
                return span;
            }
        }
        prev_is_dollar = is_token(curr, &TokenKind::Dollar);
    }
    None
}

fn is_crate_keyword(tt: &TokenTree) -> Option<Span> {
    if let TokenTree::Token(
        Token {
            kind: TokenKind::Ident(symbol, _),
            span,
        },
        _,
    ) = tt
        && symbol.as_str() == "crate"
    {
        Some(*span)
    } else {
        None
    }
}

fn is_token(tt: &TokenTree, kind: &TokenKind) -> bool {
    if let TokenTree::Token(Token { kind: other, .. }, _) = tt {
        kind == other
    } else {
        false
    }
}
