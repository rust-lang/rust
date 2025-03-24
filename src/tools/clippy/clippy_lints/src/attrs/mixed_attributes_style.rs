use super::MIXED_ATTRIBUTES_STYLE;
use clippy_utils::diagnostics::span_lint;
use rustc_ast::{AttrKind, AttrStyle, Attribute};
use rustc_data_structures::fx::FxHashSet;
use rustc_lint::{EarlyContext, LintContext};
use rustc_span::source_map::SourceMap;
use rustc_span::{SourceFile, Span, Symbol};
use std::sync::Arc;

#[derive(Hash, PartialEq, Eq)]
enum SimpleAttrKind {
    Doc,
    /// A normal attribute, with its name symbols.
    Normal(Vec<Symbol>),
}

impl From<&AttrKind> for SimpleAttrKind {
    fn from(value: &AttrKind) -> Self {
        match value {
            AttrKind::Normal(attr) => {
                let path_symbols = attr
                    .item
                    .path
                    .segments
                    .iter()
                    .map(|seg| seg.ident.name)
                    .collect::<Vec<_>>();
                Self::Normal(path_symbols)
            },
            AttrKind::DocComment(..) => Self::Doc,
        }
    }
}

pub(super) fn check(cx: &EarlyContext<'_>, item_span: Span, attrs: &[Attribute]) {
    let mut inner_attr_kind: FxHashSet<SimpleAttrKind> = FxHashSet::default();
    let mut outer_attr_kind: FxHashSet<SimpleAttrKind> = FxHashSet::default();

    let source_map = cx.sess().source_map();
    let item_src = source_map.lookup_source_file(item_span.lo());

    for attr in attrs {
        if attr.span.from_expansion() || !attr_in_same_src_as_item(source_map, &item_src, attr.span) {
            continue;
        }

        let kind: SimpleAttrKind = (&attr.kind).into();
        match attr.style {
            AttrStyle::Inner => {
                if outer_attr_kind.contains(&kind) {
                    lint_mixed_attrs(cx, attrs);
                    return;
                }
                inner_attr_kind.insert(kind);
            },
            AttrStyle::Outer => {
                if inner_attr_kind.contains(&kind) {
                    lint_mixed_attrs(cx, attrs);
                    return;
                }
                outer_attr_kind.insert(kind);
            },
        }
    }
}

fn lint_mixed_attrs(cx: &EarlyContext<'_>, attrs: &[Attribute]) {
    let mut attrs_iter = attrs.iter().filter(|attr| !attr.span.from_expansion());
    let span = if let (Some(first), Some(last)) = (attrs_iter.next(), attrs_iter.next_back()) {
        first.span.with_hi(last.span.hi())
    } else {
        return;
    };
    span_lint(
        cx,
        MIXED_ATTRIBUTES_STYLE,
        span,
        "item has both inner and outer attributes",
    );
}

fn attr_in_same_src_as_item(source_map: &SourceMap, item_src: &Arc<SourceFile>, attr_span: Span) -> bool {
    let attr_src = source_map.lookup_source_file(attr_span.lo());
    Arc::ptr_eq(item_src, &attr_src)
}
