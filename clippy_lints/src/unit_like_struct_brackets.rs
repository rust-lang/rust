use clippy_utils::{diagnostics::span_lint_and_sugg, source::snippet_opt};
use rustc_ast::ast::{Item, ItemKind, VariantData};
use rustc_errors::Applicability;
use rustc_lexer::TokenKind;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::Span;

declare_clippy_lint! {
    /// ### What it does
    /// Finds structs without fields ("unit-like structs") that are declared with brackets.
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
    #[clippy::version = "1.61.0"]
    pub UNIT_LIKE_STRUCT_BRACKETS,
    style,
    "finds struct declarations with empty brackets"
}
declare_lint_pass!(UnitLikeStructBrackets => [UNIT_LIKE_STRUCT_BRACKETS]);

impl EarlyLintPass for UnitLikeStructBrackets {
    fn check_item(&mut self, cx: &EarlyContext<'_>, item: &Item) {
        let span_after_ident = item.span.with_lo(item.ident.span.hi());

        if let ItemKind::Struct(var_data, _) = &item.kind && has_no_fields(cx, var_data, span_after_ident) {
            span_lint_and_sugg(
                cx,
                UNIT_LIKE_STRUCT_BRACKETS,
                span_after_ident,
                "found empty brackets on struct declaration",
                "remove the brackets",
                ";".to_string(),
                Applicability::MachineApplicable
            );
        }
    }
}

fn has_fields_in_hir(var_data: &VariantData) -> bool {
    match var_data {
        VariantData::Struct(defs, _) | VariantData::Tuple(defs, _) => !defs.is_empty(),
        VariantData::Unit(_) => true,
    }
}

fn has_no_ident_token(braces_span_str: &str) -> bool {
    !rustc_lexer::tokenize(braces_span_str).any(|t| t.kind == TokenKind::Ident)
}

fn has_no_fields(cx: &EarlyContext<'_>, var_data: &VariantData, braces_span: Span) -> bool {
    if has_fields_in_hir(var_data) {
        return false;
    }

    // there might still be field declarations hidden from HIR
    // (conditionaly compiled code using #[cfg(..)])

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
