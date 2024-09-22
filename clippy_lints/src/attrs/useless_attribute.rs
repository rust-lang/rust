use super::utils::{extract_clippy_lint, is_lint_level, is_word};
use super::{Attribute, USELESS_ATTRIBUTE};
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::{SpanRangeExt, first_line_of_span};
use rustc_ast::NestedMetaItem;
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
                            if let NestedMetaItem::MetaItem(meta_item) = lint
                                && meta_item.is_word()
                                && let Some(ident) = meta_item.ident()
                                && matches!(
                                    ident.name.as_str(),
                                    "ambiguous_glob_reexports"
                                        | "dead_code"
                                        | "deprecated"
                                        | "hidden_glob_reexports"
                                        | "unreachable_pub"
                                        | "unused"
                                        | "unused_braces"
                                        | "unused_import_braces"
                                        | "unused_imports"
                                )
                            {
                                return;
                            }

                            if extract_clippy_lint(lint).is_some_and(|symbol| {
                                matches!(
                                    symbol.as_str(),
                                    "wildcard_imports"
                                        | "enum_glob_use"
                                        | "redundant_pub_crate"
                                        | "macro_use_imports"
                                        | "unsafe_removed_from_name"
                                        | "module_name_repetitions"
                                        | "single_component_path_imports"
                                        | "disallowed_types"
                                        | "unused_trait_names"
                                )
                            }) {
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

                if let Some(src) = line_span.get_source_text(cx) {
                    if src.contains("#[") {
                        #[expect(clippy::collapsible_span_lint_calls)]
                        span_lint_and_then(cx, USELESS_ATTRIBUTE, line_span, "useless lint attribute", |diag| {
                            diag.span_suggestion(
                                line_span,
                                "if you just forgot a `!`, use",
                                src.replacen("#[", "#![", 1),
                                Applicability::MaybeIncorrect,
                            );
                        });
                    }
                }
            }
        }
    }
}
