//! Span maps for real files and macro expansions.

use span::Span;
use syntax::{AstNode, TextRange, ast};

pub use span::RealSpanMap;

use crate::{HirFileId, MacroCallId, db::ExpandDatabase};

pub type ExpansionSpanMap = span::SpanMap;

/// Spanmap for a macro file or a real file
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpanMap<'db> {
    /// Spanmap for a macro file
    ExpansionSpanMap(&'db ExpansionSpanMap),
    /// Spanmap for a real file
    RealSpanMap(&'db RealSpanMap),
}

impl syntax_bridge::SpanMapper for SpanMap<'_> {
    fn span_for(&self, range: TextRange) -> Span {
        self.span_for_range(range)
    }
}

impl<'db> SpanMap<'db> {
    pub fn span_for_range(&self, range: TextRange) -> Span {
        match self {
            // FIXME: Is it correct for us to only take the span at the start? This feels somewhat
            // wrong. The context will be right, but the range could be considered wrong. See
            // https://github.com/rust-lang/rust/issues/23480, we probably want to fetch the span at
            // the start and end, then merge them like rustc does in `Span::to
            Self::ExpansionSpanMap(span_map) => span_map.span_at(range.start()),
            Self::RealSpanMap(span_map) => span_map.span_for_range(range),
        }
    }

    #[inline]
    pub(crate) fn new(db: &'db dyn ExpandDatabase, file_id: HirFileId) -> SpanMap<'db> {
        match file_id {
            HirFileId::FileId(file_id) => SpanMap::RealSpanMap(db.real_span_map(file_id)),
            HirFileId::MacroFile(m) => {
                SpanMap::ExpansionSpanMap(&db.parse_macro_expansion(m).value.1)
            }
        }
    }
}

#[salsa_macros::tracked(returns(ref))]
pub(crate) fn real_span_map(
    db: &dyn ExpandDatabase,
    editioned_file_id: base_db::EditionedFileId,
) -> RealSpanMap {
    use syntax::ast::HasModuleItem;
    let mut pairs = vec![(syntax::TextSize::new(0), span::ROOT_ERASED_FILE_AST_ID)];
    let ast_id_map = db.ast_id_map(editioned_file_id.into());

    let tree = editioned_file_id.parse(db).tree();
    // This is an incrementality layer. Basically we can't use absolute ranges for our spans as that
    // would mean we'd invalidate everything whenever we type. So instead we make the text ranges
    // relative to some AstIds reducing the risk of invalidation as typing somewhere no longer
    // affects all following spans in the file.
    // There is some stuff to bear in mind here though, for one, the more "anchors" we create, the
    // easier it gets to invalidate things again as spans are as stable as their anchor's ID.
    // The other problem is proc-macros. Proc-macros have a `Span::join` api that allows them
    // to join two spans that come from the same file. rust-analyzer's proc-macro server
    // can only join two spans if they belong to the same anchor though, as the spans are relative
    // to that anchor. To do cross anchor joining we'd need to access to the ast id map to resolve
    // them again, something we might get access to in the future. But even then, proc-macros doing
    // this kind of joining makes them as stable as the AstIdMap (which is basically changing on
    // every input of the file)…

    let item_to_entry =
        |item: ast::Item| (item.syntax().text_range().start(), ast_id_map.ast_id(&item).erase());
    // Top level items make for great anchors as they are the most stable and a decent boundary
    pairs.extend(tree.items().map(item_to_entry));
    // Unfortunately, assoc items are very common in Rust, so descend into those as well and make
    // them anchors too, but only if they have no attributes attached, as those might be proc-macros
    // and using different anchors inside of them will prevent spans from being joinable.
    tree.items().for_each(|item| match &item {
        ast::Item::ExternBlock(it) if ast::attrs_including_inner(it).next().is_none() => {
            if let Some(extern_item_list) = it.extern_item_list() {
                pairs.extend(
                    extern_item_list.extern_items().map(ast::Item::from).map(item_to_entry),
                );
            }
        }
        ast::Item::Impl(it) if ast::attrs_including_inner(it).next().is_none() => {
            if let Some(assoc_item_list) = it.assoc_item_list() {
                pairs.extend(assoc_item_list.assoc_items().map(ast::Item::from).map(item_to_entry));
            }
        }
        ast::Item::Module(it) if ast::attrs_including_inner(it).next().is_none() => {
            if let Some(item_list) = it.item_list() {
                pairs.extend(item_list.items().map(item_to_entry));
            }
        }
        ast::Item::Trait(it) if ast::attrs_including_inner(it).next().is_none() => {
            if let Some(assoc_item_list) = it.assoc_item_list() {
                pairs.extend(assoc_item_list.assoc_items().map(ast::Item::from).map(item_to_entry));
            }
        }
        _ => (),
    });

    RealSpanMap::from_file(
        editioned_file_id.span_file_id(db),
        pairs.into_boxed_slice(),
        tree.syntax().text_range().end(),
    )
}

pub(crate) fn expansion_span_map(
    db: &dyn ExpandDatabase,
    file_id: MacroCallId,
) -> &ExpansionSpanMap {
    &db.parse_macro_expansion(file_id).value.1
}
