// tidy-alphabetical-start
#![allow(internal_features)]
#![feature(rustc_attrs)]
// tidy-alphabetical-end

use std::borrow::Cow;

pub use fluent_bundle::types::FluentType;
pub use fluent_bundle::{self, FluentArgs, FluentError, FluentValue};
use rustc_macros::{Decodable, Encodable};
use rustc_span::Span;
pub use unic_langid::{LanguageIdentifier, langid};

mod diagnostic_impls;
pub use diagnostic_impls::DiagArgFromDisplay;

pub fn register_functions<R, M>(bundle: &mut fluent_bundle::bundle::FluentBundle<R, M>) {
    bundle
        .add_function("STREQ", |positional, _named| match positional {
            [FluentValue::String(a), FluentValue::String(b)] => format!("{}", (a == b)).into(),
            _ => FluentValue::Error,
        })
        .expect("Failed to add a function to the bundle.");
}

/// Abstraction over a message in a diagnostic to support both translatable and non-translatable
/// diagnostic messages.
///
/// Intended to be removed once diagnostics are entirely translatable.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Encodable, Decodable)]
#[rustc_diagnostic_item = "DiagMessage"]
pub enum DiagMessage {
    /// Non-translatable diagnostic message or a message that has been translated eagerly.
    ///
    /// Some diagnostics have repeated subdiagnostics where the same interpolated variables would
    /// be instantiated multiple times with different values. These subdiagnostics' messages
    /// are translated when they are added to the parent diagnostic. This is one of the ways
    /// this variant of `DiagMessage` is produced.
    Str(Cow<'static, str>),
    /// An inline Fluent message, containing the to be translated diagnostic message.
    Inline(Cow<'static, str>),
}

impl DiagMessage {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            DiagMessage::Str(s) => Some(s),
            DiagMessage::Inline(_) => None,
        }
    }
}

impl From<String> for DiagMessage {
    fn from(s: String) -> Self {
        DiagMessage::Str(Cow::Owned(s))
    }
}
impl From<&'static str> for DiagMessage {
    fn from(s: &'static str) -> Self {
        DiagMessage::Str(Cow::Borrowed(s))
    }
}
impl From<Cow<'static, str>> for DiagMessage {
    fn from(s: Cow<'static, str>) -> Self {
        DiagMessage::Str(s)
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
    pub label: Option<DiagMessage>,
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
    span_labels: Vec<(Span, DiagMessage)>,
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

    pub fn push_primary_span(&mut self, primary_span: Span) {
        self.primary_spans.push(primary_span);
    }

    pub fn push_span_label(&mut self, span: Span, label: impl Into<DiagMessage>) {
        self.span_labels.push((span, label.into()));
    }

    pub fn push_span_diag(&mut self, span: Span, diag: DiagMessage) {
        self.span_labels.push((span, diag));
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
        !self.is_dummy()
    }

    /// Returns `true` if this contains only a dummy primary span with any hygienic context.
    pub fn is_dummy(&self) -> bool {
        self.primary_spans.iter().all(|sp| sp.is_dummy())
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

    /// Returns the span labels as contained by `MultiSpan`.
    pub fn span_labels_raw(&self) -> &[(Span, DiagMessage)] {
        &self.span_labels
    }

    /// Returns `true` if any of the span labels is displayable.
    pub fn has_span_labels(&self) -> bool {
        self.span_labels.iter().any(|(sp, _)| !sp.is_dummy())
    }

    /// Clone this `MultiSpan` without keeping any of the span labels - sometimes a `MultiSpan` is
    /// to be re-used in another diagnostic, but includes `span_labels` which have translated
    /// messages. These translated messages would fail to translate without their diagnostic
    /// arguments which are unlikely to be cloned alongside the `Span`.
    pub fn clone_ignoring_labels(&self) -> Self {
        Self { primary_spans: self.primary_spans.clone(), ..MultiSpan::new() }
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

fn icu_locale_from_unic_langid(lang: LanguageIdentifier) -> Option<icu_locale::Locale> {
    icu_locale::Locale::try_from_str(&lang.to_string()).ok()
}

pub fn fluent_value_from_str_list_sep_by_and(l: Vec<Cow<'_, str>>) -> FluentValue<'_> {
    // Fluent requires 'static value here for its AnyEq usages.
    #[derive(Clone, PartialEq, Debug)]
    struct FluentStrListSepByAnd(Vec<String>);

    impl FluentType for FluentStrListSepByAnd {
        fn duplicate(&self) -> Box<dyn FluentType + Send> {
            Box::new(self.clone())
        }

        fn as_string(&self, intls: &intl_memoizer::IntlLangMemoizer) -> Cow<'static, str> {
            let result = intls
                .with_try_get::<MemoizableListFormatter, _, _>((), |list_formatter| {
                    list_formatter.format_to_string(self.0.iter())
                })
                .unwrap();
            Cow::Owned(result)
        }

        fn as_string_threadsafe(
            &self,
            intls: &intl_memoizer::concurrent::IntlLangMemoizer,
        ) -> Cow<'static, str> {
            let result = intls
                .with_try_get::<MemoizableListFormatter, _, _>((), |list_formatter| {
                    list_formatter.format_to_string(self.0.iter())
                })
                .unwrap();
            Cow::Owned(result)
        }
    }

    struct MemoizableListFormatter(icu_list::ListFormatter);

    impl std::ops::Deref for MemoizableListFormatter {
        type Target = icu_list::ListFormatter;
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl intl_memoizer::Memoizable for MemoizableListFormatter {
        type Args = ();
        type Error = ();

        fn construct(lang: LanguageIdentifier, _args: Self::Args) -> Result<Self, Self::Error>
        where
            Self: Sized,
        {
            let locale = icu_locale_from_unic_langid(lang)
                .unwrap_or_else(|| rustc_baked_icu_data::supported_locales::EN);
            let list_formatter = icu_list::ListFormatter::try_new_and_unstable(
                &rustc_baked_icu_data::BakedDataProvider,
                locale.into(),
                icu_list::options::ListFormatterOptions::default()
                    .with_length(icu_list::options::ListLength::Wide),
            )
            .expect("Failed to create list formatter");

            Ok(MemoizableListFormatter(list_formatter))
        }
    }

    let l = l.into_iter().map(|x| x.into_owned()).collect();

    FluentValue::Custom(Box::new(FluentStrListSepByAnd(l)))
}

/// Simplified version of `FluentArg` that can implement `Encodable` and `Decodable`. Collection of
/// `DiagArg` are converted to `FluentArgs` (consuming the collection) at the start of diagnostic
/// emission.
pub type DiagArg<'iter> = (&'iter DiagArgName, &'iter DiagArgValue);

/// Name of a diagnostic argument.
pub type DiagArgName = Cow<'static, str>;

/// Simplified version of `FluentValue` that can implement `Encodable` and `Decodable`. Converted
/// to a `FluentValue` by the emitter to be used in diagnostic translation.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Encodable, Decodable)]
pub enum DiagArgValue {
    Str(Cow<'static, str>),
    // This gets converted to a `FluentNumber`, which is an `f64`. An `i32`
    // safely fits in an `f64`. Any integers bigger than that will be converted
    // to strings in `into_diag_arg` and stored using the `Str` variant.
    Number(i32),
    StrListSepByAnd(Vec<Cow<'static, str>>),
}

/// Converts a value of a type into a `DiagArg` (typically a field of an `Diag` struct).
/// Implemented as a custom trait rather than `From` so that it is implemented on the type being
/// converted rather than on `DiagArgValue`, which enables types from other `rustc_*` crates to
/// implement this.
pub trait IntoDiagArg {
    /// Convert `Self` into a `DiagArgValue` suitable for rendering in a diagnostic.
    ///
    /// It takes a `path` where "long values" could be written to, if the `DiagArgValue` is too big
    /// for displaying on the terminal. This path comes from the `Diag` itself. When rendering
    /// values that come from `TyCtxt`, like `Ty<'_>`, they can use `TyCtxt::short_string`. If a
    /// value has no shortening logic that could be used, the argument can be safely ignored.
    fn into_diag_arg(self, path: &mut Option<std::path::PathBuf>) -> DiagArgValue;
}

impl IntoDiagArg for DiagArgValue {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        self
    }
}

impl From<DiagArgValue> for FluentValue<'static> {
    fn from(val: DiagArgValue) -> Self {
        match val {
            DiagArgValue::Str(s) => From::from(s),
            DiagArgValue::Number(n) => From::from(n),
            DiagArgValue::StrListSepByAnd(l) => fluent_value_from_str_list_sep_by_and(l),
        }
    }
}
