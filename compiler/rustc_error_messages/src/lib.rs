use rustc_data_structures::sync::Lrc;
use rustc_macros::{Decodable, Encodable};
use rustc_span::Span;
use std::borrow::Cow;
use tracing::debug;

pub use fluent::{FluentArgs, FluentValue};

static FALLBACK_FLUENT_RESOURCE: &'static str = include_str!("../locales/en-US/diagnostics.ftl");

pub type FluentBundle = fluent::FluentBundle<fluent::FluentResource>;

/// Return the default `FluentBundle` with standard en-US diagnostic messages.
pub fn fallback_fluent_bundle() -> Lrc<FluentBundle> {
    let fallback_resource = fluent::FluentResource::try_new(FALLBACK_FLUENT_RESOURCE.to_string())
        .expect("failed to parse ftl resource");
    debug!(?fallback_resource);
    let mut fallback_bundle = FluentBundle::new(vec![unic_langid::langid!("en-US")]);
    fallback_bundle.add_resource(fallback_resource).expect("failed to add resource to bundle");
    let fallback_bundle = Lrc::new(fallback_bundle);
    fallback_bundle
}

/// Identifier for the Fluent message/attribute corresponding to a diagnostic message.
type FluentId = Cow<'static, str>;

/// Abstraction over a message in a diagnostic to support both translatable and non-translatable
/// diagnostic messages.
///
/// Intended to be removed once diagnostics are entirely translatable.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Encodable, Decodable)]
pub enum DiagnosticMessage {
    /// Non-translatable diagnostic message.
    // FIXME(davidtwco): can a `Cow<'static, str>` be used here?
    Str(String),
    /// Identifier for a Fluent message corresponding to the diagnostic message.
    FluentIdentifier(FluentId, Option<FluentId>),
}

impl DiagnosticMessage {
    /// Returns the `String` contained within the `DiagnosticMessage::Str` variant, assuming that
    /// this diagnostic message is of the legacy, non-translatable variety. Panics if this
    /// assumption does not hold.
    ///
    /// Don't use this - it exists to support some places that do comparison with diagnostic
    /// strings.
    pub fn expect_str(&self) -> &str {
        match self {
            DiagnosticMessage::Str(s) => s,
            _ => panic!("expected non-translatable diagnostic message"),
        }
    }

    /// Create a `DiagnosticMessage` for the provided Fluent identifier.
    pub fn fluent(id: impl Into<Cow<'static, str>>) -> Self {
        DiagnosticMessage::FluentIdentifier(id.into(), None)
    }

    /// Create a `DiagnosticMessage` for the provided Fluent identifier and attribute.
    pub fn fluent_attr(
        id: impl Into<Cow<'static, str>>,
        attr: impl Into<Cow<'static, str>>,
    ) -> Self {
        DiagnosticMessage::FluentIdentifier(id.into(), Some(attr.into()))
    }
}

/// `From` impl that enables existing diagnostic calls to functions which now take
/// `impl Into<DiagnosticMessage>` to continue to work as before.
impl<S: Into<String>> From<S> for DiagnosticMessage {
    fn from(s: S) -> Self {
        DiagnosticMessage::Str(s.into())
    }
}

/// A span together with some additional data.
#[derive(Clone, Debug)]
pub struct SpanLabel {
    /// The span we are going to include in the final snippet.
    pub span: Span,

    /// Is this a primary span? This is the "locus" of the message,
    /// and is indicated with a `^^^^` underline, versus `----`.
    pub is_primary: bool,

    /// What label should we attach to this span (if any)?
    pub label: Option<DiagnosticMessage>,
}

/// A collection of `Span`s.
///
/// Spans have two orthogonal attributes:
///
/// - They can be *primary spans*. In this case they are the locus of
///   the error, and would be rendered with `^^^`.
/// - They can have a *label*. In this case, the label is written next
///   to the mark in the snippet when we render.
#[derive(Clone, Debug, Hash, PartialEq, Eq, Encodable, Decodable)]
pub struct MultiSpan {
    primary_spans: Vec<Span>,
    span_labels: Vec<(Span, DiagnosticMessage)>,
}

impl MultiSpan {
    #[inline]
    pub fn new() -> MultiSpan {
        MultiSpan { primary_spans: vec![], span_labels: vec![] }
    }

    pub fn from_span(primary_span: Span) -> MultiSpan {
        MultiSpan { primary_spans: vec![primary_span], span_labels: vec![] }
    }

    pub fn from_spans(mut vec: Vec<Span>) -> MultiSpan {
        vec.sort();
        MultiSpan { primary_spans: vec, span_labels: vec![] }
    }

    pub fn push_span_label(&mut self, span: Span, label: impl Into<DiagnosticMessage>) {
        self.span_labels.push((span, label.into()));
    }

    /// Selects the first primary span (if any).
    pub fn primary_span(&self) -> Option<Span> {
        self.primary_spans.first().cloned()
    }

    /// Returns all primary spans.
    pub fn primary_spans(&self) -> &[Span] {
        &self.primary_spans
    }

    /// Returns `true` if any of the primary spans are displayable.
    pub fn has_primary_spans(&self) -> bool {
        self.primary_spans.iter().any(|sp| !sp.is_dummy())
    }

    /// Returns `true` if this contains only a dummy primary span with any hygienic context.
    pub fn is_dummy(&self) -> bool {
        let mut is_dummy = true;
        for span in &self.primary_spans {
            if !span.is_dummy() {
                is_dummy = false;
            }
        }
        is_dummy
    }

    /// Replaces all occurrences of one Span with another. Used to move `Span`s in areas that don't
    /// display well (like std macros). Returns whether replacements occurred.
    pub fn replace(&mut self, before: Span, after: Span) -> bool {
        let mut replacements_occurred = false;
        for primary_span in &mut self.primary_spans {
            if *primary_span == before {
                *primary_span = after;
                replacements_occurred = true;
            }
        }
        for span_label in &mut self.span_labels {
            if span_label.0 == before {
                span_label.0 = after;
                replacements_occurred = true;
            }
        }
        replacements_occurred
    }

    /// Returns the strings to highlight. We always ensure that there
    /// is an entry for each of the primary spans -- for each primary
    /// span `P`, if there is at least one label with span `P`, we return
    /// those labels (marked as primary). But otherwise we return
    /// `SpanLabel` instances with empty labels.
    pub fn span_labels(&self) -> Vec<SpanLabel> {
        let is_primary = |span| self.primary_spans.contains(&span);

        let mut span_labels = self
            .span_labels
            .iter()
            .map(|&(span, ref label)| SpanLabel {
                span,
                is_primary: is_primary(span),
                label: Some(label.clone()),
            })
            .collect::<Vec<_>>();

        for &span in &self.primary_spans {
            if !span_labels.iter().any(|sl| sl.span == span) {
                span_labels.push(SpanLabel { span, is_primary: true, label: None });
            }
        }

        span_labels
    }

    /// Returns `true` if any of the span labels is displayable.
    pub fn has_span_labels(&self) -> bool {
        self.span_labels.iter().any(|(sp, _)| !sp.is_dummy())
    }
}

impl From<Span> for MultiSpan {
    fn from(span: Span) -> MultiSpan {
        MultiSpan::from_span(span)
    }
}

impl From<Vec<Span>> for MultiSpan {
    fn from(spans: Vec<Span>) -> MultiSpan {
        MultiSpan::from_spans(spans)
    }
}
