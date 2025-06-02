use clippy_config::Conf;
use clippy_utils::diagnostics::span_lint_hir_and_then;
use core::mem::replace;
use rustc_errors::Applicability;
use rustc_hir::{HirId, Item, ItemKind};
use rustc_lint::{LateContext, LateLintPass, LintContext};
use rustc_session::impl_lint_pass;
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
    /// ```no_run
    /// struct HTTPResponse;
    /// ```
    /// Use instead:
    /// ```no_run
    /// struct HttpResponse;
    /// ```
    #[clippy::version = "1.51.0"]
    pub UPPER_CASE_ACRONYMS,
    style,
    "capitalized acronyms are against the naming convention"
}

pub struct UpperCaseAcronyms {
    avoid_breaking_exported_api: bool,
    upper_case_acronyms_aggressive: bool,
}

impl UpperCaseAcronyms {
    pub fn new(conf: &'static Conf) -> Self {
        Self {
            avoid_breaking_exported_api: conf.avoid_breaking_exported_api,
            upper_case_acronyms_aggressive: conf.upper_case_acronyms_aggressive,
        }
    }
}

impl_lint_pass!(UpperCaseAcronyms => [UPPER_CASE_ACRONYMS]);

fn contains_acronym(s: &str) -> bool {
    let mut count = 0;
    for c in s.chars() {
        if c.is_ascii_uppercase() {
            count += 1;
            if count == 3 {
                return true;
            }
        } else {
            count = 0;
        }
    }
    count == 2
}

fn check_ident(cx: &LateContext<'_>, ident: &Ident, hir_id: HirId, be_aggressive: bool) {
    let s = ident.as_str();

    // By default, only warn for upper case identifiers with at least 3 characters.
    let replacement = if s.len() > 2 && s.bytes().all(|c| c.is_ascii_uppercase()) {
        let mut r = String::with_capacity(s.len());
        let mut s = s.chars();
        r.push(s.next().unwrap());
        r.extend(s.map(|c| c.to_ascii_lowercase()));
        r
    } else if be_aggressive
        // Only lint if the ident starts with an upper case character.
        && let unprefixed = s.trim_start_matches('_')
        && unprefixed.starts_with(|c: char| c.is_ascii_uppercase())
        && contains_acronym(unprefixed)
    {
        let mut r = String::with_capacity(s.len());
        let mut s = s.chars();
        let mut prev_upper = false;
        while let Some(c) = s.next() {
            r.push(
                if replace(&mut prev_upper, c.is_ascii_uppercase())
                    && s.clone().next().is_none_or(|c| c.is_ascii_uppercase())
                {
                    c.to_ascii_lowercase()
                } else {
                    c
                },
            );
        }
        r
    } else {
        return;
    };

    span_lint_hir_and_then(
        cx,
        UPPER_CASE_ACRONYMS,
        hir_id,
        ident.span,
        format!("name `{ident}` contains a capitalized acronym"),
        |diag| {
            diag.span_suggestion(
                ident.span,
                "consider making the acronym lowercase, except the initial letter",
                replacement,
                Applicability::MaybeIncorrect,
            );
        },
    );
}

impl LateLintPass<'_> for UpperCaseAcronyms {
    fn check_item(&mut self, cx: &LateContext<'_>, it: &Item<'_>) {
        // do not lint public items or in macros
        if it.span.in_external_macro(cx.sess().source_map())
            || (self.avoid_breaking_exported_api && cx.effective_visibilities.is_exported(it.owner_id.def_id))
        {
            return;
        }
        match it.kind {
            ItemKind::TyAlias(ident, ..) | ItemKind::Struct(ident, ..) | ItemKind::Trait(_, _, ident, ..) => {
                check_ident(cx, &ident, it.hir_id(), self.upper_case_acronyms_aggressive);
            },
            ItemKind::Enum(ident, _, ref enumdef) => {
                check_ident(cx, &ident, it.hir_id(), self.upper_case_acronyms_aggressive);
                // check enum variants separately because again we only want to lint on private enums and
                // the fn check_variant does not know about the vis of the enum of its variants
                enumdef.variants.iter().for_each(|variant| {
                    check_ident(cx, &variant.ident, variant.hir_id, self.upper_case_acronyms_aggressive);
                });
            },
            _ => {},
        }
    }
}
