use super::DUPLICATED_ATTRIBUTES;
use super::utils::is_lint_level;
use clippy_utils::diagnostics::span_lint_and_then;
use rustc_ast::{Attribute, MetaItem};
use rustc_ast_pretty::pprust::path_to_string;
use rustc_data_structures::fx::FxHashMap;
use rustc_lint::EarlyContext;
use rustc_span::{Span, Symbol, sym};
use std::collections::hash_map::Entry;

#[derive(Clone, Copy)]
enum Context {
    /// The meta item is an entire attribute.
    Attribute,
    /// The meta item is an argument of the lint level attribute `level`,
    /// e.g. a lint name or `reason = "..."`.
    LintName { level: Symbol },
}

fn emit_if_duplicated(
    cx: &EarlyContext<'_>,
    meta: &MetaItem,
    seen: &mut FxHashMap<String, Span>,
    complete_path: String,
) {
    match seen.entry(complete_path) {
        Entry::Vacant(v) => {
            v.insert(meta.span);
        },
        Entry::Occupied(o) => {
            span_lint_and_then(cx, DUPLICATED_ATTRIBUTES, meta.span, "duplicated attribute", |diag| {
                diag.span_note(*o.get(), "first defined here");
                diag.span_help(meta.span, "remove this attribute");
            });
        },
    }
}

fn check_duplicated_attr(cx: &EarlyContext<'_>, meta: &MetaItem, seen: &mut FxHashMap<String, Span>, ctx: Context) {
    if meta.span.from_expansion() {
        return;
    }
    match ctx {
        Context::Attribute => {
            // Only attributes whose semantics are known here are checked, currently these
            // are the lint level attributes, whose arguments are independent lint names.
            if let Some(ident) = meta.ident()
                && is_lint_level(ident.name)
                && let Some(items) = meta.meta_item_list()
            {
                for item in items {
                    if let Some(item) = item.meta_item() {
                        check_duplicated_attr(cx, item, seen, Context::LintName { level: ident.name });
                    }
                }
            }
        },
        Context::LintName { level } => {
            // Multiple lint level attributes may share the same `reason`
            if meta.has_name(sym::reason) {
                return;
            }
            emit_if_duplicated(cx, meta, seen, format!("{level}:{}", path_to_string(&meta.path)));
        },
    }
}

pub fn check(cx: &EarlyContext<'_>, attrs: &[Attribute]) {
    let mut seen = FxHashMap::default();

    for attr in attrs {
        if let Some(meta) = attr.meta() {
            check_duplicated_attr(cx, &meta, &mut seen, Context::Attribute);
        }
    }
}
