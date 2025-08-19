use super::USELESS_ATTRIBUTE;
use super::utils::{is_lint_level, is_word, namespace_and_lint};
use clippy_utils::diagnostics::span_lint_and_then;
use clippy_utils::source::{SpanRangeExt, first_line_of_span};
use clippy_utils::sym;
use rustc_ast::{Attribute, Item, ItemKind};
use rustc_errors::Applicability;
use rustc_lint::{EarlyContext, LintContext};

pub(super) fn check(cx: &EarlyContext<'_>, item: &Item, attrs: &[Attribute]) {
    let skip_unused_imports = attrs.iter().any(|attr| attr.has_name(sym::macro_use));

    for attr in attrs {
        if attr.span.in_external_macro(cx.sess().source_map()) {
            return;
        }
        if let Some(lint_list) = &attr.meta_item_list()
            && attr.ident().is_some_and(|ident| is_lint_level(ident.name, attr.id))
        {
            for lint in lint_list {
                match item.kind {
                    ItemKind::Use(..) => {
                        let (namespace @ (Some(sym::clippy) | None), Some(name)) = namespace_and_lint(lint) else {
                            return;
                        };

                        if namespace.is_none()
                            && matches!(
                                name,
                                sym::ambiguous_glob_reexports
                                    | sym::dead_code
                                    | sym::deprecated
                                    | sym::hidden_glob_reexports
                                    | sym::unreachable_pub
                                    | sym::unused
                                    | sym::unused_braces
                                    | sym::unused_import_braces
                                    | sym::unused_imports
                                    | sym::redundant_imports
                            )
                        {
                            return;
                        }

                        if namespace == Some(sym::clippy)
                            && matches!(
                                name,
                                sym::wildcard_imports
                                    | sym::enum_glob_use
                                    | sym::redundant_pub_crate
                                    | sym::macro_use_imports
                                    | sym::unsafe_removed_from_name
                                    | sym::module_name_repetitions
                                    | sym::single_component_path_imports
                                    | sym::disallowed_types
                                    | sym::unused_trait_names
                            )
                        {
                            return;
                        }
                    },
                    ItemKind::ExternCrate(..) => {
                        if is_word(lint, sym::unused_imports) && skip_unused_imports {
                            return;
                        }
                        if is_word(lint, sym::unused_extern_crates) {
                            return;
                        }
                    },
                    _ => {},
                }
            }
            let line_span = first_line_of_span(cx, attr.span);

            if let Some(src) = line_span.get_source_text(cx)
                && src.contains("#[")
            {
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
