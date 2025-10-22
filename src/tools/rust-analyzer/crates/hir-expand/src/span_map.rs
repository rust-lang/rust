//! Span maps for real files and macro expansions.

use span::{Span, SyntaxContext};
use stdx::TupleExt;
use syntax::{AstNode, TextRange, ast};
use triomphe::Arc;

pub use span::RealSpanMap;

use crate::{HirFileId, MacroCallId, attrs::collect_attrs, db::ExpandDatabase};

pub type ExpansionSpanMap = span::SpanMap<SyntaxContext>;

/// Spanmap for a macro file or a real file
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SpanMap {
    /// Spanmap for a macro file
    ExpansionSpanMap(Arc<ExpansionSpanMap>),
    /// Spanmap for a real file
    RealSpanMap(Arc<RealSpanMap>),
}

#[derive(Copy, Clone)]
pub enum SpanMapRef<'a> {
    /// Spanmap for a macro file
    ExpansionSpanMap(&'a ExpansionSpanMap),
    /// Spanmap for a real file
    RealSpanMap(&'a RealSpanMap),
}

impl syntax_bridge::SpanMapper<Span> for SpanMap {
    fn span_for(&self, range: TextRange) -> Span {
        self.span_for_range(range)
    }
}

impl syntax_bridge::SpanMapper<Span> for SpanMapRef<'_> {
    fn span_for(&self, range: TextRange) -> Span {
        self.span_for_range(range)
    }
}

impl SpanMap {
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

    pub fn as_ref(&self) -> SpanMapRef<'_> {
        match self {
            Self::ExpansionSpanMap(span_map) => SpanMapRef::ExpansionSpanMap(span_map),
            Self::RealSpanMap(span_map) => SpanMapRef::RealSpanMap(span_map),
        }
    }

    #[inline]
    pub(crate) fn new(db: &dyn ExpandDatabase, file_id: HirFileId) -> SpanMap {
        match file_id {
            HirFileId::FileId(file_id) => SpanMap::RealSpanMap(db.real_span_map(file_id)),
            HirFileId::MacroFile(m) => {
                SpanMap::ExpansionSpanMap(db.parse_macro_expansion(m).value.1)
            }
        }
    }
}

impl SpanMapRef<'_> {
    pub fn span_for_range(self, range: TextRange) -> Span {
        match self {
            Self::ExpansionSpanMap(span_map) => span_map.span_at(range.start()),
            Self::RealSpanMap(span_map) => span_map.span_for_range(range),
        }
    }
}

pub(crate) fn real_span_map(
    db: &dyn ExpandDatabase,
    editioned_file_id: base_db::EditionedFileId,
) -> Arc<RealSpanMap> {
    use syntax::ast::HasModuleItem;
    let mut pairs = vec![(syntax::TextSize::new(0), span::ROOT_ERASED_FILE_AST_ID)];
    let ast_id_map = db.ast_id_map(editioned_file_id.into());

    let tree = db.parse(editioned_file_id).tree();
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
        ast::Item::ExternBlock(it)
            if !collect_attrs(it).map(TupleExt::tail).any(|it| it.is_left()) =>
        {
            if let Some(extern_item_list) = it.extern_item_list() {
                pairs.extend(
                    extern_item_list.extern_items().map(ast::Item::from).map(item_to_entry),
                );
            }
        }
        ast::Item::Impl(it) if !collect_attrs(it).map(TupleExt::tail).any(|it| it.is_left()) => {
            if let Some(assoc_item_list) = it.assoc_item_list() {
                pairs.extend(assoc_item_list.assoc_items().map(ast::Item::from).map(item_to_entry));
            }
        }
        ast::Item::Module(it) if !collect_attrs(it).map(TupleExt::tail).any(|it| it.is_left()) => {
            if let Some(item_list) = it.item_list() {
                pairs.extend(item_list.items().map(item_to_entry));
            }
        }
        ast::Item::Trait(it) if !collect_attrs(it).map(TupleExt::tail).any(|it| it.is_left()) => {
            if let Some(assoc_item_list) = it.assoc_item_list() {
                pairs.extend(assoc_item_list.assoc_items().map(ast::Item::from).map(item_to_entry));
            }
        }
        _ => (),
    });

    Arc::new(RealSpanMap::from_file(
        editioned_file_id.editioned_file_id(db),
        pairs.into_boxed_slice(),
        tree.syntax().text_range().end(),
    ))
}

pub(crate) fn expansion_span_map(
    db: &dyn ExpandDatabase,
    file_id: MacroCallId,
) -> Arc<ExpansionSpanMap> {
    db.parse_macro_expansion(file_id).value.1
}
