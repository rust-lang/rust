use crate::utils::span_lint_and_sugg;
use if_chain::if_chain;
use itertools::Itertools;
use rustc_ast::ast::{Item, ItemKind, Variant};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, EarlyLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_lint_pass, declare_tool_lint};
use rustc_span::symbol::Ident;

declare_clippy_lint! {
    /// **What it does:** Checks for camel case name containing a capitalized acronym.
    ///
    /// **Why is this bad?** In CamelCase, acronyms count as one word.
    /// See [naming conventions](https://rust-lang.github.io/api-guidelines/naming.html#casing-conforms-to-rfc-430-c-case)
    /// for more.
    ///
    /// **Known problems:** When two acronyms are contiguous, the lint can't tell where
    /// the first acronym ends and the second starts, so it suggests to lowercase all of
    /// the letters in the second acronym.
    ///
    /// **Example:**
    ///
    /// ```rust
    /// struct HTTPResponse;
    /// ```
    /// Use instead:
    /// ```rust
    /// struct HttpResponse;
    /// ```
    pub UPPER_CASE_ACRONYMS,
    style,
    "capitalized acronyms are against the naming convention"
}

declare_lint_pass!(UpperCaseAcronyms => [UPPER_CASE_ACRONYMS]);

fn correct_ident(ident: &str) -> String {
    let ident = ident.chars().rev().collect::<String>();
    let fragments = ident
        .split_inclusive(|x: char| !x.is_ascii_lowercase())
        .rev()
        .map(|x| x.chars().rev().collect::<String>());

    let mut ident = fragments.clone().next().unwrap();
    for (ref prev, ref curr) in fragments.tuple_windows() {
        if [prev, curr]
            .iter()
            .all(|s| s.len() == 1 && s.chars().next().unwrap().is_ascii_uppercase())
        {
            ident.push_str(&curr.to_ascii_lowercase());
        } else {
            ident.push_str(curr);
        }
    }
    ident
}

fn check_ident(cx: &EarlyContext<'_>, ident: &Ident) {
    let span = ident.span;
    let ident = &ident.as_str();
    let corrected = correct_ident(ident);
    if ident != &corrected {
        span_lint_and_sugg(
            cx,
            UPPER_CASE_ACRONYMS,
            span,
            &format!("name `{}` contains a capitalized acronym", ident),
            "consider making the acronym lowercase, except the initial letter",
            corrected,
            Applicability::MaybeIncorrect,
        )
    }
}

impl EarlyLintPass for UpperCaseAcronyms {
    fn check_item(&mut self, cx: &EarlyContext<'_>, it: &Item) {
        if_chain! {
            if !in_external_macro(cx.sess(), it.span);
            if matches!(
                it.kind,
                ItemKind::TyAlias(..) | ItemKind::Enum(..) | ItemKind::Struct(..) | ItemKind::Trait(..)
            );
            then {
                check_ident(cx, &it.ident);
            }
        }
    }

    fn check_variant(&mut self, cx: &EarlyContext<'_>, v: &Variant) {
        check_ident(cx, &v.ident);
    }
}
