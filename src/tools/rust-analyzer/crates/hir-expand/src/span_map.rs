//! Span maps for real files and macro expansions.
use span::{FileId, HirFileId, HirFileIdRepr, MacroFileId, Span};
use syntax::{AstNode, TextRange};
use triomphe::Arc;

pub use span::RealSpanMap;

use crate::db::ExpandDatabase;

pub type ExpansionSpanMap = span::SpanMap<Span>;

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

impl mbe::SpanMapper<Span> for SpanMap {
    fn span_for(&self, range: TextRange) -> Span {
        self.span_for_range(range)
    }
}

impl mbe::SpanMapper<Span> for SpanMapRef<'_> {
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
        match file_id.repr() {
            HirFileIdRepr::FileId(file_id) => SpanMap::RealSpanMap(db.real_span_map(file_id)),
            HirFileIdRepr::MacroFile(m) => {
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

pub(crate) fn real_span_map(db: &dyn ExpandDatabase, file_id: FileId) -> Arc<RealSpanMap> {
    use syntax::ast::HasModuleItem;
    let mut pairs = vec![(syntax::TextSize::new(0), span::ROOT_ERASED_FILE_AST_ID)];
    let ast_id_map = db.ast_id_map(file_id.into());
    let tree = db.parse(file_id).tree();
    // FIXME: Descend into modules and other item containing items that are not annotated with attributes
    // and allocate pairs for those as well. This gives us finer grained span anchors resulting in
    // better incrementality
    pairs.extend(
        tree.items()
            .map(|item| (item.syntax().text_range().start(), ast_id_map.ast_id(&item).erase())),
    );

    Arc::new(RealSpanMap::from_file(
        file_id,
        pairs.into_boxed_slice(),
        tree.syntax().text_range().end(),
    ))
}

pub(crate) fn expansion_span_map(
    db: &dyn ExpandDatabase,
    file_id: MacroFileId,
) -> Arc<ExpansionSpanMap> {
    db.parse_macro_expansion(file_id).value.1
}
