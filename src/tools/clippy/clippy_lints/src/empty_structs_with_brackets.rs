use clippy_utils::{diagnostics::span_lint_and_then, source::snippet_opt};
use rustc_ast::ast::{Item, ItemKind, VariantData};
use rustc_errors::Applicability;
use rustc_lexer::TokenKind;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Finds structs without fields (a so-called "empty struct") that are declared with brackets.
    ///
    /// ### Why is this bad?
    /// Empty brackets after a struct declaration can be omitted.
    ///
    /// ### Example
    /// ```rust
    /// struct Cookie {}
    /// ```
    /// Use instead:
    /// ```rust
    /// struct Cookie;
    /// ```
    #[clippy::version = "1.62.0"]
    pub EMPTY_STRUCTS_WITH_BRACKETS,
    restriction,
    "finds struct declarations with empty brackets"
}
declare_lint_pass!(EmptyStructsWithBrackets => [EMPTY_STRUCTS_WITH_BRACKETS]);

impl EarlyLintPass for EmptyStructsWithBrackets {
    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &Item) {
        let span_after_ident = item.span.with_lo(item.ident.span.hi());

        if let ItemKind::Struct(var_data, _) = &item.kind
            && has_brackets(var_data)
            && has_no_fields(cx, var_data, span_after_ident) {
            span_lint_and_then(
                cx,
                EMPTY_STRUCTS_WITH_BRACKETS,
                span_after_ident,
                "found empty brackets on struct declaration",
                |diagnostic| {
                    diagnostic.span_suggestion_hidden(
                        span_after_ident,
                        "remove the brackets",
                        ";",
                        Applicability::Unspecified);
                    },
            );
        }
    }
}

fn has_no_ident_token(braces_span_str: &str) -> bool {
    !rustc_lexer::tokenize(braces_span_str).any(|t| t.kind == TokenKind::Ident)
}

fn has_brackets(var_data: &VariantData) -> bool {
    !matches!(var_data, VariantData::Unit(_))
}

fn has_no_fields(cx: &EarlyContext<'_>, var_data: &VariantData, braces_span: Span) -> bool {
    if !var_data.fields().is_empty() {
        return false;
    }

    // there might still be field declarations hidden from the AST
    // (conditionally compiled code using #[cfg(..)])

    let Some(braces_span_str) = snippet_opt(cx, braces_span) else {
        return false;
    };

    has_no_ident_token(braces_span_str.as_ref())
}

#[cfg(test)]
mod unit_test {
    use super::*;

    #[test]
    fn test_has_no_ident_token() {
        let input = "{ field: u8 }";
        assert!(!has_no_ident_token(input));

        let input = "(u8, String);";
        assert!(!has_no_ident_token(input));

        let input = " {
                // test = 5
        }
        ";
        assert!(has_no_ident_token(input));

        let input = " ();";
        assert!(has_no_ident_token(input));
    }
}
