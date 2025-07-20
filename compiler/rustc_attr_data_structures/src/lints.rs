use rustc_macros::HashStable_Generic;
use rustc_span::Span;

#[derive(Clone, Debug, HashStable_Generic)]
pub struct AttributeLint<Id> {
    pub id: Id,
    pub span: Span,
    pub kind: AttributeLintKind,
}

#[derive(Clone, Debug, HashStable_Generic)]
pub enum AttributeLintKind {
    UnusedDuplicate { this: Span, other: Span, warning: bool },
    IllFormedAttributeInput { suggestions: Vec<String> },
    EmptyAttribute { first_span: Span },
    UnsafeAttrOutsideUnsafe { attribute_name_span: Span, sugg_spans: (Span, Span) },
}
