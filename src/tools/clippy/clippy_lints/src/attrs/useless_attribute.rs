use super::utils::{extract_clippy_lint, is_lint_level, is_word};
use super::{Attribute, USELESS_ATTRIBUTE};
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::{first_line_of_span, snippet_opt};
use rustc_errors::Applicability;
use rustc_hir::{Item, ItemKind};
use rustc_lint::{LateContext, LintContext};
use rustc_middle::lint::in_external_macro;
use rustc_span::sym;

pub(super) fn check(cx: &LateContext<'_>, item: &Item<'_>, attrs: &[Attribute]) {
    let skip_unused_imports = attrs.iter().any(|attr| attr.has_name(sym::macro_use));

    for attr in attrs {
        if in_external_macro(cx.sess(), attr.span) {
            return;
        }
        if let Some(lint_list) = &attr.meta_item_list() {
            if attr.ident().map_or(false, |ident| is_lint_level(ident.name, attr.id)) {
                for lint in lint_list {
                    match item.kind {
                        ItemKind::Use(..) => {
                            if is_word(lint, sym::unused_imports)
                                || is_word(lint, sym::deprecated)
                                || is_word(lint, sym!(unreachable_pub))
                                || is_word(lint, sym!(unused))
                                || is_word(lint, sym!(unused_import_braces))
                                || extract_clippy_lint(lint).map_or(false, |s| {
                                    matches!(
                                        s.as_str(),
                                        "wildcard_imports"
                                            | "enum_glob_use"
                                            | "redundant_pub_crate"
                                            | "macro_use_imports"
                                            | "unsafe_removed_from_name"
                                            | "module_name_repetitions"
                                            | "single_component_path_imports"
                                    )
                                })
                            {
                                return;
                            }
                        },
                        ItemKind::ExternCrate(..) => {
                            if is_word(lint, sym::unused_imports) && skip_unused_imports {
                                return;
                            }
                            if is_word(lint, sym!(unused_extern_crates)) {
                                return;
                            }
                        },
                        _ => {},
                    }
                }
                let line_span = first_line_of_span(cx, attr.span);

                if let Some(mut sugg) = snippet_opt(cx, line_span) {
                    if sugg.contains("#[") {
                        span_lint_and_then(cx, USELESS_ATTRIBUTE, line_span, "useless lint attribute", |diag| {
                            sugg = sugg.replacen("#[", "#![", 1);
                            diag.span_suggestion(
                                line_span,
                                "if you just forgot a `!`, use",
                                sugg,
                                Applicability::MaybeIncorrect,
                            );
                        });
                    }
                }
            }
        }
    }
}
