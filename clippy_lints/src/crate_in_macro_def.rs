use clippy_utils::diagnostics::span_lint_and_sugg;
use rustc_ast::ast::MacroDef;
use rustc_ast::node_id::NodeId;
use rustc_ast::token::{Token, TokenKind};
use rustc_ast::tokenstream::{TokenStream, TokenTree};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for use of `crate` as opposed to `$crate` in a macro definition.
    ///
    /// ### Why is this bad?
    /// `crate` refers to macro call's crate, whereas `$crate` refers to the macro
    /// definition's crate. Rarely is the former intended. See:
    /// https://doc.rust-lang.org/reference/macros-by-example.html#hygiene
    ///
    /// ### Example
    /// ```rust
    /// macro_rules! print_message {
    ///     () => {
    ///         println!("{}", crate::MESSAGE);
    ///     };
    /// }
    /// pub const MESSAGE: &str = "Hello!";
    /// ```
    /// Use instead:
    /// ```rust
    /// macro_rules! print_message {
    ///     () => {
    ///         println!("{}", $crate::MESSAGE);
    ///     };
    /// }
    /// pub const MESSAGE: &str = "Hello!";
    /// ```
    #[clippy::version = "1.61.0"]
    pub CRATE_IN_MACRO_DEF,
    correctness,
    "using `crate` in a macro definition"
}
declare_lint_pass!(CrateInMacroDef => [CRATE_IN_MACRO_DEF]);

impl EarlyLintPass for CrateInMacroDef {
    fn check_mac_def(&mut self, cx: &EarlyContext<'_>, macro_def: &MacroDef, _: NodeId) {
        let tts = macro_def.body.inner_tokens();
        if let Some(span) = contains_unhygienic_crate_reference(&tts) {
            span_lint_and_sugg(
                cx,
                CRATE_IN_MACRO_DEF,
                span,
                "reference to the macro call's crate, which is rarely intended",
                "if reference to the macro definition's crate is intended, use",
                String::from("$crate"),
                Applicability::MachineApplicable,
            );
        }
    }
}

fn contains_unhygienic_crate_reference(tts: &TokenStream) -> Option<Span> {
    let mut prev_is_dollar = false;
    let mut cursor = tts.trees();
    while let Some(curr) = cursor.next() {
        if_chain! {
            if !prev_is_dollar;
            if let Some(span) = is_crate_keyword(&curr);
            if let Some(next) = cursor.look_ahead(0);
            if is_token(next, &TokenKind::ModSep);
            then {
                return Some(span);
            }
        }
        if let TokenTree::Delimited(_, _, tts) = &curr {
            let span = contains_unhygienic_crate_reference(tts);
            if span.is_some() {
                return span;
            }
        }
        prev_is_dollar = is_token(&curr, &TokenKind::Dollar);
    }
    None
}

fn is_crate_keyword(tt: &TokenTree) -> Option<Span> {
    if_chain! {
        if let TokenTree::Token(Token { kind: TokenKind::Ident(symbol, _), span }) = tt;
        if symbol.as_str() == "crate";
        then { Some(*span) } else { None }
    }
}

fn is_token(tt: &TokenTree, kind: &TokenKind) -> bool {
    if let TokenTree::Token(Token { kind: other, .. }) = tt {
        kind == other
    } else {
        false
    }
}
