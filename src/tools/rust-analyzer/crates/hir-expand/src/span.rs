//! Spanmaps allow turning absolute ranges into relative ranges for incrementality purposes as well
//! as associating spans with text ranges in a particular file.
use base_db::{
    span::{ErasedFileAstId, SpanAnchor, SpanData, SyntaxContextId, ROOT_ERASED_FILE_AST_ID},
    FileId,
};
use syntax::{ast::HasModuleItem, AstNode, TextRange, TextSize};
use triomphe::Arc;

use crate::db::ExpandDatabase;

pub type ExpansionSpanMap = mbe::SpanMap<SpanData>;

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

impl mbe::SpanMapper<SpanData> for SpanMap {
    fn span_for(&self, range: TextRange) -> SpanData {
        self.span_for_range(range)
    }
}
impl mbe::SpanMapper<SpanData> for SpanMapRef<'_> {
    fn span_for(&self, range: TextRange) -> SpanData {
        self.span_for_range(range)
    }
}
impl mbe::SpanMapper<SpanData> for RealSpanMap {
    fn span_for(&self, range: TextRange) -> SpanData {
        self.span_for_range(range)
    }
}

impl SpanMap {
    pub fn span_for_range(&self, range: TextRange) -> SpanData {
        match self {
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
}

impl SpanMapRef<'_> {
    pub fn span_for_range(self, range: TextRange) -> SpanData {
        match self {
            Self::ExpansionSpanMap(span_map) => span_map.span_at(range.start()),
            Self::RealSpanMap(span_map) => span_map.span_for_range(range),
        }
    }
}

#[derive(PartialEq, Eq, Hash, Debug)]
pub struct RealSpanMap {
    file_id: FileId,
    /// Invariant: Sorted vec over TextSize
    // FIXME: SortedVec<(TextSize, ErasedFileAstId)>?
    pairs: Box<[(TextSize, ErasedFileAstId)]>,
    end: TextSize,
}

impl RealSpanMap {
    /// Creates a real file span map that returns absolute ranges (relative ranges to the root ast id).
    pub fn absolute(file_id: FileId) -> Self {
        RealSpanMap {
            file_id,
            pairs: Box::from([(TextSize::new(0), ROOT_ERASED_FILE_AST_ID)]),
            end: TextSize::new(!0),
        }
    }

    pub fn from_file(db: &dyn ExpandDatabase, file_id: FileId) -> Self {
        let mut pairs = vec![(TextSize::new(0), ROOT_ERASED_FILE_AST_ID)];
        let ast_id_map = db.ast_id_map(file_id.into());
        let tree = db.parse(file_id).tree();
        pairs
            .extend(tree.items().map(|item| {
                (item.syntax().text_range().start(), ast_id_map.ast_id(&item).erase())
            }));
        RealSpanMap {
            file_id,
            pairs: pairs.into_boxed_slice(),
            end: tree.syntax().text_range().end(),
        }
    }

    pub fn span_for_range(&self, range: TextRange) -> SpanData {
        assert!(
            range.end() <= self.end,
            "range {range:?} goes beyond the end of the file {:?}",
            self.end
        );
        let start = range.start();
        let idx = self
            .pairs
            .binary_search_by(|&(it, _)| it.cmp(&start).then(std::cmp::Ordering::Less))
            .unwrap_err();
        let (offset, ast_id) = self.pairs[idx - 1];
        SpanData {
            range: range - offset,
            anchor: SpanAnchor { file_id: self.file_id, ast_id },
            ctx: SyntaxContextId::ROOT,
        }
    }
}
