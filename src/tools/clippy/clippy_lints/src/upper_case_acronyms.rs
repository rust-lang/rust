use clippy_utils::diagnostics::span_lint_and_sugg;
use itertools::Itertools;
use rustc_errors::Applicability;
use rustc_hir::{Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_session::{declare_tool_lint, impl_lint_pass};
use rustc_span::symbol::Ident;

declare_clippy_lint! {
    /// ### What it does
    /// Checks for fully capitalized names and optionally names containing a capitalized acronym.
    ///
    /// ### Why is this bad?
    /// In CamelCase, acronyms count as one word.
    /// See [naming conventions](https://rust-lang.github.io/api-guidelines/naming.html#casing-conforms-to-rfc-430-c-case)
    /// for more.
    ///
    /// By default, the lint only triggers on fully-capitalized names.
    /// You can use the `upper-case-acronyms-aggressive: true` config option to enable linting
    /// on all camel case names
    ///
    /// ### Known problems
    /// When two acronyms are contiguous, the lint can't tell where
    /// the first acronym ends and the second starts, so it suggests to lowercase all of
    /// the letters in the second acronym.
    ///
    /// ### Example
    /// ```rust
    /// struct HTTPResponse;
    /// ```
    /// Use instead:
    /// ```rust
    /// struct HttpResponse;
    /// ```
    #[clippy::version = "1.51.0"]
    pub UPPER_CASE_ACRONYMS,
    style,
    "capitalized acronyms are against the naming convention"
}

#[derive(Default)]
pub struct UpperCaseAcronyms {
    avoid_breaking_exported_api: bool,
    upper_case_acronyms_aggressive: bool,
}

impl UpperCaseAcronyms {
    pub fn new(avoid_breaking_exported_api: bool, aggressive: bool) -> Self {
        Self {
            avoid_breaking_exported_api,
            upper_case_acronyms_aggressive: aggressive,
        }
    }
}

impl_lint_pass!(UpperCaseAcronyms => [UPPER_CASE_ACRONYMS]);

fn correct_ident(ident: &str) -> String {
    let ident = ident.chars().rev().collect::<String>();
    let fragments = ident
        .split_inclusive(|x: char| !x.is_ascii_lowercase())
        .rev()
        .map(|x| x.chars().rev().collect::<String>());

    let mut ident = fragments.clone().next().unwrap();
    for (ref prev, ref curr) in fragments.tuple_windows() {
        if <[&String; 2]>::from((prev, curr))
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

fn check_ident(cx: &LateContext<'_>, ident: &Ident, be_aggressive: bool) {
    let span = ident.span;
    let ident = ident.as_str();
    let corrected = correct_ident(ident);
    // warn if we have pure-uppercase idents
    // assume that two-letter words are some kind of valid abbreviation like FP for false positive
    // (and don't warn)
    if (ident.chars().all(|c| c.is_ascii_uppercase()) && ident.len() > 2)
    // otherwise, warn if we have SOmeTHING lIKE THIs but only warn with the aggressive
    // upper-case-acronyms-aggressive config option enabled
    || (be_aggressive && ident != corrected)
    {
        span_lint_and_sugg(
            cx,
            UPPER_CASE_ACRONYMS,
            span,
            &format!("name `{ident}` contains a capitalized acronym"),
            "consider making the acronym lowercase, except the initial letter",
            corrected,
            Applicability::MaybeIncorrect,
        );
    }
}

impl LateLintPass<'_> for UpperCaseAcronyms {
    fn check_item(&mut self, cx: &LateContext<'_>, it: &Item<'_>) {
        // do not lint public items or in macros
        if in_external_macro(cx.sess(), it.span)
            || (self.avoid_breaking_exported_api && cx.effective_visibilities.is_exported(it.owner_id.def_id))
        {
            return;
        }
        match it.kind {
            ItemKind::TyAlias(..) | ItemKind::Struct(..) | ItemKind::Trait(..) => {
                check_ident(cx, &it.ident, self.upper_case_acronyms_aggressive);
            },
            ItemKind::Enum(ref enumdef, _) => {
                check_ident(cx, &it.ident, self.upper_case_acronyms_aggressive);
                // check enum variants separately because again we only want to lint on private enums and
                // the fn check_variant does not know about the vis of the enum of its variants
                enumdef
                    .variants
                    .iter()
                    .for_each(|variant| check_ident(cx, &variant.ident, self.upper_case_acronyms_aggressive));
            },
            _ => {},
        }
    }
}
