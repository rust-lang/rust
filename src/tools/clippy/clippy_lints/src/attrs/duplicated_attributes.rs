use super::DUPLICATED_ATTRIBUTES;
use clippy_utils::diagnostics::span_lint_and_then;
use rustc_ast::{MetaItem, MetaItemInner};
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::Attribute;
use rustc_lint::LateContext;
use rustc_span::symbol::Ident;
use rustc_span::{Span, Symbol, sym};
use std::collections::hash_map::Entry;
use thin_vec::ThinVec;

fn emit_if_duplicated(
    cx: &LateContext<'_>,
    span: Span,
    attr_paths: &mut FxHashMap<String, Span>,
    complete_path: String,
) {
    match attr_paths.entry(complete_path) {
        Entry::Vacant(v) => {
            v.insert(span);
        },
        Entry::Occupied(o) => {
            span_lint_and_then(cx, DUPLICATED_ATTRIBUTES, span, "duplicated attribute", |diag| {
                diag.span_note(*o.get(), "first defined here");
                diag.span_help(span, "remove this attribute");
            });
        },
    }
}

trait AttrOrMetaItem {
    fn ident(&self) -> Option<Ident>;
    fn span(&self) -> Span;
    fn meta_item_list(&self) -> Option<ThinVec<MetaItemInner>>;
    fn value_str(&self) -> Option<Symbol>;
}

impl AttrOrMetaItem for Attribute {
    fn ident(&self) -> Option<Ident> {
        rustc_ast::attr::AttributeExt::ident(self)
    }

    fn span(&self) -> Span {
        rustc_ast::attr::AttributeExt::span(self)
    }

    fn meta_item_list(&self) -> Option<ThinVec<MetaItemInner>> {
        rustc_ast::attr::AttributeExt::meta_item_list(self)
    }

    fn value_str(&self) -> Option<Symbol> {
        rustc_ast::attr::AttributeExt::value_str(self)
    }
}

impl AttrOrMetaItem for MetaItem {
    fn ident(&self) -> Option<Ident> {
        MetaItem::ident(self)
    }

    fn span(&self) -> Span {
        self.span
    }

    fn meta_item_list(&self) -> Option<ThinVec<MetaItemInner>> {
        MetaItem::meta_item_list(self).map(|i| i.iter().cloned().collect())
    }

    fn value_str(&self) -> Option<Symbol> {
        MetaItem::value_str(self)
    }
}

fn check_duplicated_attr(
    cx: &LateContext<'_>,
    attr: &impl AttrOrMetaItem,
    attr_paths: &mut FxHashMap<String, Span>,
    parent: &mut Vec<String>,
) {
    if attr.span().from_expansion() {
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
        emit_if_duplicated(
            cx,
            attr.span(),
            attr_paths,
            format!("{}:{name}={value}", parent.join(":")),
        );
    } else if let Some(sub_attrs) = attr.meta_item_list() {
        parent.push(name.as_str().to_string());
        for sub_attr in sub_attrs {
            if let Some(meta) = sub_attr.meta_item() {
                check_duplicated_attr(cx, meta, attr_paths, parent);
            }
        }
        parent.pop();
    } else {
        emit_if_duplicated(cx, attr.span(), attr_paths, format!("{}:{name}", parent.join(":")));
    }
}

pub fn check(cx: &LateContext<'_>, attrs: &[Attribute]) {
    let mut attr_paths = FxHashMap::default();

    for attr in attrs {
        if !rustc_ast::attr::AttributeExt::is_doc_comment(attr) {
            check_duplicated_attr(cx, attr, &mut attr_paths, &mut Vec::new());
        }
    }
}
