use clippy_utils::diagnostics::span_lint_and_sugg;
use rustc_ast::ast::{Item, ItemKind, VariantData};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::{declare_lint_pass, declare_tool_lint};

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
        if let ItemKind::Struct(var_data, _) = &item.kind && has_no_fields(var_data) {
            let span_after_ident = item.span.with_lo(item.ident.span.hi());

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

fn has_no_fields(var_data: &VariantData) -> bool {
    match var_data {
        VariantData::Struct(defs, _) | VariantData::Tuple(defs, _) => defs.is_empty(),
        VariantData::Unit(_) => false,
    }
}
