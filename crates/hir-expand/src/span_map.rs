//! Span maps for real files and macro expansions.
use span::Span;
use syntax::TextRange;
use triomphe::Arc;

pub use span::RealSpanMap;

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
    pub fn span_for_range(self, range: TextRange) -> Span {
        match self {
            Self::ExpansionSpanMap(span_map) => span_map.span_at(range.start()),
            Self::RealSpanMap(span_map) => span_map.span_for_range(range),
        }
    }
}
