use super::DUPLICATED_ATTRIBUTES;
use clippy_utils::diagnostics::span_lint_and_then;
use rustc_ast::{Attribute, MetaItem};
use rustc_data_structures::fx::FxHashMap;
use rustc_lint::EarlyContext;
use rustc_span::{Span, sym};
use std::collections::hash_map::Entry;

fn emit_if_duplicated(
    cx: &EarlyContext<'_>,
    attr: &MetaItem,
    attr_paths: &mut FxHashMap<String, Span>,
    complete_path: String,
) {
    match attr_paths.entry(complete_path) {
        Entry::Vacant(v) => {
            v.insert(attr.span);
        },
        Entry::Occupied(o) => {
            span_lint_and_then(cx, DUPLICATED_ATTRIBUTES, attr.span, "duplicated attribute", |diag| {
                diag.span_note(*o.get(), "first defined here");
                diag.span_help(attr.span, "remove this attribute");
            });
        },
    }
}

fn check_duplicated_attr(
    cx: &EarlyContext<'_>,
    attr: &MetaItem,
    attr_paths: &mut FxHashMap<String, Span>,
    parent: &mut Vec<String>,
) {
    if attr.span.from_expansion() {
        return;
    }
    let Some(ident) = attr.ident() else { return };
    let name = ident.name;
    if name == sym::doc || name == sym::cfg_attr || name == sym::rustc_on_unimplemented || name == sym::reason {
        // FIXME: Would be nice to handle `cfg_attr` as well. Only problem is to check that cfg
        // conditions are the same.
        // `#[rustc_on_unimplemented]` contains duplicated subattributes, that's expected.
        return;
    }
    if let Some(direct_parent) = parent.last()
        && ["cfg", "cfg_attr"].contains(&direct_parent.as_str())
        && [sym::all, sym::not, sym::any].contains(&name)
    {
        // FIXME: We don't correctly check `cfg`s for now, so if it's more complex than just a one
        // level `cfg`, we leave.
        return;
    }
    if let Some(value) = attr.value_str() {
        emit_if_duplicated(cx, attr, attr_paths, format!("{}:{name}={value}", parent.join(":")));
    } else if let Some(sub_attrs) = attr.meta_item_list() {
        parent.push(name.as_str().to_string());
        for sub_attr in sub_attrs {
            if let Some(meta) = sub_attr.meta_item() {
                check_duplicated_attr(cx, meta, attr_paths, parent);
            }
        }
        parent.pop();
    } else {
        emit_if_duplicated(cx, attr, attr_paths, format!("{}:{name}", parent.join(":")));
    }
}

pub fn check(cx: &EarlyContext<'_>, attrs: &[Attribute]) {
    let mut attr_paths = FxHashMap::default();

    for attr in attrs {
        if let Some(meta) = attr.meta() {
            check_duplicated_attr(cx, &meta, &mut attr_paths, &mut Vec::new());
        }
    }
}
