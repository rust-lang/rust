use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::snippet_opt;
use rustc_ast::ast::{Item, ItemKind, Variant, VariantData};
use rustc_errors::Applicability;
use rustc_lexer::TokenKind;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::declare_lint_pass;
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Finds structs without fields (a so-called "empty struct") that are declared with brackets.
    ///
    /// ### Why restrict this?
    /// Empty brackets after a struct declaration can be omitted,
    /// and it may be desirable to do so consistently for style.
    ///
    /// However, removing the brackets also introduces a public constant named after the struct,
    /// so this is not just a syntactic simplification but an API change, and adding them back
    /// is a *breaking* API change.
    ///
    /// ### Example
    /// ```no_run
    /// struct Cookie {}
    /// struct Biscuit();
    /// ```
    /// Use instead:
    /// ```no_run
    /// struct Cookie;
    /// struct Biscuit;
    /// ```
    #[clippy::version = "1.62.0"]
    pub EMPTY_STRUCTS_WITH_BRACKETS,
    restriction,
    "finds struct declarations with empty brackets"
}

declare_clippy_lint! {
    /// ### What it does
    /// Finds enum variants without fields that are declared with empty brackets.
    ///
    /// ### Why restrict this?
    /// Empty brackets after a enum variant declaration are redundant and can be omitted,
    /// and it may be desirable to do so consistently for style.
    ///
    /// However, removing the brackets also introduces a public constant named after the variant,
    /// so this is not just a syntactic simplification but an API change, and adding them back
    /// is a *breaking* API change.
    ///
    /// ### Example
    /// ```no_run
    /// enum MyEnum {
    ///     HasData(u8),
    ///     HasNoData(),       // redundant parentheses
    ///     NoneHereEither {}, // redundant braces
    /// }
    /// ```
    ///
    /// Use instead:
    /// ```no_run
    /// enum MyEnum {
    ///     HasData(u8),
    ///     HasNoData,
    ///     NoneHereEither,
    /// }
    /// ```
    #[clippy::version = "1.77.0"]
    pub EMPTY_ENUM_VARIANTS_WITH_BRACKETS,
    restriction,
    "finds enum variants with empty brackets"
}

declare_lint_pass!(EmptyWithBrackets => [EMPTY_STRUCTS_WITH_BRACKETS, EMPTY_ENUM_VARIANTS_WITH_BRACKETS]);

impl EarlyLintPass for EmptyWithBrackets {
    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &Item) {
        if let ItemKind::Struct(ident, var_data, _) = &item.kind
            && has_brackets(var_data)
            && let span_after_ident = item.span.with_lo(ident.span.hi())
            && has_no_fields(cx, var_data, span_after_ident)
        {
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
                        Applicability::Unspecified,
                    );
                },
            );
        }
    }

    fn check_variant(&mut self, cx: &EarlyContext<'_>, variant: &Variant) {
        let span_after_ident = variant.span.with_lo(variant.ident.span.hi());

        if has_brackets(&variant.data) && has_no_fields(cx, &variant.data, span_after_ident) {
            span_lint_and_then(
                cx,
                EMPTY_ENUM_VARIANTS_WITH_BRACKETS,
                span_after_ident,
                "enum variant has empty brackets",
                |diagnostic| {
                    diagnostic.span_suggestion_hidden(
                        span_after_ident,
                        "remove the brackets",
                        "",
                        Applicability::MaybeIncorrect,
                    );
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
