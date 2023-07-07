#![feature(let_chains)]
#![feature(lazy_cell)]
#![feature(rustc_attrs)]
#![feature(type_alias_impl_trait)]
#![deny(rustc::untranslatable_diagnostic)]
#![deny(rustc::diagnostic_outside_of_impl)]

#[macro_use]
extern crate tracing;

use fluent_bundle::FluentResource;
use fluent_syntax::parser::ParserError;
use icu_provider_adapters::fallback::{LocaleFallbackProvider, LocaleFallbacker};
use rustc_data_structures::sync::{IntoDynSyncSend, Lrc};
use rustc_fluent_macro::fluent_messages;
use rustc_macros::{Decodable, Encodable};
use rustc_span::Span;
use std::borrow::Cow;
use std::error::Error;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

#[cfg(not(parallel_compiler))]
use std::cell::LazyCell as Lazy;
#[cfg(parallel_compiler)]
use std::sync::LazyLock as Lazy;

#[cfg(parallel_compiler)]
use intl_memoizer::concurrent::IntlLangMemoizer;
#[cfg(not(parallel_compiler))]
use intl_memoizer::IntlLangMemoizer;

pub use fluent_bundle::{self, types::FluentType, FluentArgs, FluentError, FluentValue};
pub use unic_langid::{langid, LanguageIdentifier};

fluent_messages! { "../messages.ftl" }

pub type FluentBundle =
    IntoDynSyncSend<fluent_bundle::bundle::FluentBundle<FluentResource, IntlLangMemoizer>>;

#[cfg(not(parallel_compiler))]
fn new_bundle(locales: Vec<LanguageIdentifier>) -> FluentBundle {
    IntoDynSyncSend(fluent_bundle::bundle::FluentBundle::new(locales))
}

#[cfg(parallel_compiler)]
fn new_bundle(locales: Vec<LanguageIdentifier>) -> FluentBundle {
    IntoDynSyncSend(fluent_bundle::bundle::FluentBundle::new_concurrent(locales))
}

#[derive(Debug)]
pub enum TranslationBundleError {
    /// Failed to read from `.ftl` file.
    ReadFtl(io::Error),
    /// Failed to parse contents of `.ftl` file.
    ParseFtl(ParserError),
    /// Failed to add `FluentResource` to `FluentBundle`.
    AddResource(FluentError),
    /// `$sysroot/share/locale/$locale` does not exist.
    MissingLocale,
    /// Cannot read directory entries of `$sysroot/share/locale/$locale`.
    ReadLocalesDir(io::Error),
    /// Cannot read directory entry of `$sysroot/share/locale/$locale`.
    ReadLocalesDirEntry(io::Error),
    /// `$sysroot/share/locale/$locale` is not a directory.
    LocaleIsNotDir,
}

impl fmt::Display for TranslationBundleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TranslationBundleError::ReadFtl(e) => write!(f, "could not read ftl file: {}", e),
            TranslationBundleError::ParseFtl(e) => {
                write!(f, "could not parse ftl file: {}", e)
            }
            TranslationBundleError::AddResource(e) => write!(f, "failed to add resource: {}", e),
            TranslationBundleError::MissingLocale => write!(f, "missing locale directory"),
            TranslationBundleError::ReadLocalesDir(e) => {
                write!(f, "could not read locales dir: {}", e)
            }
            TranslationBundleError::ReadLocalesDirEntry(e) => {
                write!(f, "could not read locales dir entry: {}", e)
            }
            TranslationBundleError::LocaleIsNotDir => {
                write!(f, "`$sysroot/share/locales/$locale` is not a directory")
            }
        }
    }
}

impl Error for TranslationBundleError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            TranslationBundleError::ReadFtl(e) => Some(e),
            TranslationBundleError::ParseFtl(e) => Some(e),
            TranslationBundleError::AddResource(e) => Some(e),
            TranslationBundleError::MissingLocale => None,
            TranslationBundleError::ReadLocalesDir(e) => Some(e),
            TranslationBundleError::ReadLocalesDirEntry(e) => Some(e),
            TranslationBundleError::LocaleIsNotDir => None,
        }
    }
}

impl From<(FluentResource, Vec<ParserError>)> for TranslationBundleError {
    fn from((_, mut errs): (FluentResource, Vec<ParserError>)) -> Self {
        TranslationBundleError::ParseFtl(errs.pop().expect("failed ftl parse with no errors"))
    }
}

impl From<Vec<FluentError>> for TranslationBundleError {
    fn from(mut errs: Vec<FluentError>) -> Self {
        TranslationBundleError::AddResource(
            errs.pop().expect("failed adding resource to bundle with no errors"),
        )
    }
}

/// Returns Fluent bundle with the user's locale resources from
/// `$sysroot/share/locale/$requested_locale/*.ftl`.
///
/// If `-Z additional-ftl-path` was provided, load that resource and add it  to the bundle
/// (overriding any conflicting messages).
#[instrument(level = "trace")]
pub fn fluent_bundle(
    mut user_provided_sysroot: Option<PathBuf>,
    mut sysroot_candidates: Vec<PathBuf>,
    requested_locale: Option<LanguageIdentifier>,
    additional_ftl_path: Option<&Path>,
    with_directionality_markers: bool,
) -> Result<Option<Lrc<FluentBundle>>, TranslationBundleError> {
    if requested_locale.is_none() && additional_ftl_path.is_none() {
        return Ok(None);
    }

    let fallback_locale = langid!("en-US");
    let requested_fallback_locale = requested_locale.as_ref() == Some(&fallback_locale);
    trace!(?requested_fallback_locale);
    if requested_fallback_locale && additional_ftl_path.is_none() {
        return Ok(None);
    }
    // If there is only `-Z additional-ftl-path`, assume locale is "en-US", otherwise use user
    // provided locale.
    let locale = requested_locale.clone().unwrap_or(fallback_locale);
    trace!(?locale);
    let mut bundle = new_bundle(vec![locale]);

    // Add convenience functions available to ftl authors.
    register_functions(&mut bundle);

    // Fluent diagnostics can insert directionality isolation markers around interpolated variables
    // indicating that there may be a shift from right-to-left to left-to-right text (or
    // vice-versa). These are disabled because they are sometimes visible in the error output, but
    // may be worth investigating in future (for example: if type names are left-to-right and the
    // surrounding diagnostic messages are right-to-left, then these might be helpful).
    bundle.set_use_isolating(with_directionality_markers);

    // If the user requests the default locale then don't try to load anything.
    if let Some(requested_locale) = requested_locale {
        let mut found_resources = false;
        for sysroot in user_provided_sysroot.iter_mut().chain(sysroot_candidates.iter_mut()) {
            sysroot.push("share");
            sysroot.push("locale");
            sysroot.push(requested_locale.to_string());
            trace!(?sysroot);

            if !sysroot.exists() {
                trace!("skipping");
                continue;
            }

            if !sysroot.is_dir() {
                return Err(TranslationBundleError::LocaleIsNotDir);
            }

            for entry in sysroot.read_dir().map_err(TranslationBundleError::ReadLocalesDir)? {
                let entry = entry.map_err(TranslationBundleError::ReadLocalesDirEntry)?;
                let path = entry.path();
                trace!(?path);
                if path.extension().and_then(|s| s.to_str()) != Some("ftl") {
                    trace!("skipping");
                    continue;
                }

                let resource_str =
                    fs::read_to_string(path).map_err(TranslationBundleError::ReadFtl)?;
                let resource =
                    FluentResource::try_new(resource_str).map_err(TranslationBundleError::from)?;
                trace!(?resource);
                bundle.add_resource(resource).map_err(TranslationBundleError::from)?;
                found_resources = true;
            }
        }

        if !found_resources {
            return Err(TranslationBundleError::MissingLocale);
        }
    }

    if let Some(additional_ftl_path) = additional_ftl_path {
        let resource_str =
            fs::read_to_string(additional_ftl_path).map_err(TranslationBundleError::ReadFtl)?;
        let resource =
            FluentResource::try_new(resource_str).map_err(TranslationBundleError::from)?;
        trace!(?resource);
        bundle.add_resource_overriding(resource);
    }

    let bundle = Lrc::new(bundle);
    Ok(Some(bundle))
}

fn register_functions(bundle: &mut FluentBundle) {
    bundle
        .add_function("STREQ", |positional, _named| match positional {
            [FluentValue::String(a), FluentValue::String(b)] => format!("{}", (a == b)).into(),
            _ => FluentValue::Error,
        })
        .expect("Failed to add a function to the bundle.");
}

/// Type alias for the result of `fallback_fluent_bundle` - a reference-counted pointer to a lazily
/// evaluated fluent bundle.
pub type LazyFallbackBundle = Lrc<Lazy<FluentBundle, impl FnOnce() -> FluentBundle>>;

/// Return the default `FluentBundle` with standard "en-US" diagnostic messages.
#[instrument(level = "trace", skip(resources))]
pub fn fallback_fluent_bundle(
    resources: Vec<&'static str>,
    with_directionality_markers: bool,
) -> LazyFallbackBundle {
    Lrc::new(Lazy::new(move || {
        let mut fallback_bundle = new_bundle(vec![langid!("en-US")]);

        register_functions(&mut fallback_bundle);

        // See comment in `fluent_bundle`.
        fallback_bundle.set_use_isolating(with_directionality_markers);

        for resource in resources {
            let resource = FluentResource::try_new(resource.to_string())
                .expect("failed to parse fallback fluent resource");
            fallback_bundle.add_resource_overriding(resource);
        }

        fallback_bundle
    }))
}

/// Identifier for the Fluent message/attribute corresponding to a diagnostic message.
type FluentId = Cow<'static, str>;

/// Abstraction over a message in a subdiagnostic (i.e. label, note, help, etc) to support both
/// translatable and non-translatable diagnostic messages.
///
/// Translatable messages for subdiagnostics are typically attributes attached to a larger Fluent
/// message so messages of this type must be combined with a `DiagnosticMessage` (using
/// `DiagnosticMessage::with_subdiagnostic_message`) before rendering. However, subdiagnostics from
/// the `Subdiagnostic` derive refer to Fluent identifiers directly.
#[rustc_diagnostic_item = "SubdiagnosticMessage"]
pub enum SubdiagnosticMessage {
    /// Non-translatable diagnostic message.
    Str(Cow<'static, str>),
    /// Translatable message which has already been translated eagerly.
    ///
    /// Some diagnostics have repeated subdiagnostics where the same interpolated variables would
    /// be instantiated multiple times with different values. As translation normally happens
    /// immediately prior to emission, after the diagnostic and subdiagnostic derive logic has run,
    /// the setting of diagnostic arguments in the derived code will overwrite previous variable
    /// values and only the final value will be set when translation occurs - resulting in
    /// incorrect diagnostics. Eager translation results in translation for a subdiagnostic
    /// happening immediately after the subdiagnostic derive's logic has been run. This variant
    /// stores messages which have been translated eagerly.
    Eager(Cow<'static, str>),
    /// Identifier of a Fluent message. Instances of this variant are generated by the
    /// `Subdiagnostic` derive.
    FluentIdentifier(FluentId),
    /// Attribute of a Fluent message. Needs to be combined with a Fluent identifier to produce an
    /// actual translated message. Instances of this variant are generated by the `fluent_messages`
    /// macro.
    ///
    /// <https://projectfluent.org/fluent/guide/attributes.html>
    FluentAttr(FluentId),
}

impl From<String> for SubdiagnosticMessage {
    fn from(s: String) -> Self {
        SubdiagnosticMessage::Str(Cow::Owned(s))
    }
}
impl From<&'static str> for SubdiagnosticMessage {
    fn from(s: &'static str) -> Self {
        SubdiagnosticMessage::Str(Cow::Borrowed(s))
    }
}
impl From<Cow<'static, str>> for SubdiagnosticMessage {
    fn from(s: Cow<'static, str>) -> Self {
        SubdiagnosticMessage::Str(s)
    }
}

/// Abstraction over a message in a diagnostic to support both translatable and non-translatable
/// diagnostic messages.
///
/// Intended to be removed once diagnostics are entirely translatable.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Encodable, Decodable)]
#[rustc_diagnostic_item = "DiagnosticMessage"]
pub enum DiagnosticMessage {
    /// Non-translatable diagnostic message.
    Str(Cow<'static, str>),
    /// Translatable message which has already been translated eagerly.
    ///
    /// Some diagnostics have repeated subdiagnostics where the same interpolated variables would
    /// be instantiated multiple times with different values. As translation normally happens
    /// immediately prior to emission, after the diagnostic and subdiagnostic derive logic has run,
    /// the setting of diagnostic arguments in the derived code will overwrite previous variable
    /// values and only the final value will be set when translation occurs - resulting in
    /// incorrect diagnostics. Eager translation results in translation for a subdiagnostic
    /// happening immediately after the subdiagnostic derive's logic has been run. This variant
    /// stores messages which have been translated eagerly.
    Eager(Cow<'static, str>),
    /// Identifier for a Fluent message (with optional attribute) corresponding to the diagnostic
    /// message.
    ///
    /// <https://projectfluent.org/fluent/guide/hello.html>
    /// <https://projectfluent.org/fluent/guide/attributes.html>
    FluentIdentifier(FluentId, Option<FluentId>),
}

impl DiagnosticMessage {
    /// Given a `SubdiagnosticMessage` which may contain a Fluent attribute, create a new
    /// `DiagnosticMessage` that combines that attribute with the Fluent identifier of `self`.
    ///
    /// - If the `SubdiagnosticMessage` is non-translatable then return the message as a
    /// `DiagnosticMessage`.
    /// - If `self` is non-translatable then return `self`'s message.
    pub fn with_subdiagnostic_message(&self, sub: SubdiagnosticMessage) -> Self {
        let attr = match sub {
            SubdiagnosticMessage::Str(s) => return DiagnosticMessage::Str(s),
            SubdiagnosticMessage::Eager(s) => return DiagnosticMessage::Eager(s),
            SubdiagnosticMessage::FluentIdentifier(id) => {
                return DiagnosticMessage::FluentIdentifier(id, None);
            }
            SubdiagnosticMessage::FluentAttr(attr) => attr,
        };

        match self {
            DiagnosticMessage::Str(s) => DiagnosticMessage::Str(s.clone()),
            DiagnosticMessage::Eager(s) => DiagnosticMessage::Eager(s.clone()),
            DiagnosticMessage::FluentIdentifier(id, _) => {
                DiagnosticMessage::FluentIdentifier(id.clone(), Some(attr))
            }
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            DiagnosticMessage::Eager(s) | DiagnosticMessage::Str(s) => Some(s),
            DiagnosticMessage::FluentIdentifier(_, _) => None,
        }
    }
}

impl From<String> for DiagnosticMessage {
    fn from(s: String) -> Self {
        DiagnosticMessage::Str(Cow::Owned(s))
    }
}
impl From<&'static str> for DiagnosticMessage {
    fn from(s: &'static str) -> Self {
        DiagnosticMessage::Str(Cow::Borrowed(s))
    }
}
impl From<Cow<'static, str>> for DiagnosticMessage {
    fn from(s: Cow<'static, str>) -> Self {
        DiagnosticMessage::Str(s)
    }
}

/// A workaround for "good path" ICEs when formatting types in disabled lints.
///
/// Delays formatting until `.into(): DiagnosticMessage` is used.
pub struct DelayDm<F>(pub F);

impl<F: FnOnce() -> String> From<DelayDm<F>> for DiagnosticMessage {
    fn from(DelayDm(f): DelayDm<F>) -> Self {
        DiagnosticMessage::from(f())
    }
}

/// Translating *into* a subdiagnostic message from a diagnostic message is a little strange - but
/// the subdiagnostic functions (e.g. `span_label`) take a `SubdiagnosticMessage` and the
/// subdiagnostic derive refers to typed identifiers that are `DiagnosticMessage`s, so need to be
/// able to convert between these, as much as they'll be converted back into `DiagnosticMessage`
/// using `with_subdiagnostic_message` eventually. Don't use this other than for the derive.
impl Into<SubdiagnosticMessage> for DiagnosticMessage {
    fn into(self) -> SubdiagnosticMessage {
        match self {
            DiagnosticMessage::Str(s) => SubdiagnosticMessage::Str(s),
            DiagnosticMessage::Eager(s) => SubdiagnosticMessage::Eager(s),
            DiagnosticMessage::FluentIdentifier(id, None) => {
                SubdiagnosticMessage::FluentIdentifier(id)
            }
            // There isn't really a sensible behaviour for this because it loses information but
            // this is the most sensible of the behaviours.
            DiagnosticMessage::FluentIdentifier(_, Some(attr)) => {
                SubdiagnosticMessage::FluentAttr(attr)
            }
        }
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

    pub fn pop_span_label(&mut self) -> Option<(Span, DiagnosticMessage)> {
        self.span_labels.pop()
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

fn icu_locale_from_unic_langid(lang: LanguageIdentifier) -> Option<icu_locid::Locale> {
    icu_locid::Locale::try_from_bytes(lang.to_string().as_bytes()).ok()
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

        #[cfg(not(parallel_compiler))]
        fn as_string_threadsafe(
            &self,
            _intls: &intl_memoizer::concurrent::IntlLangMemoizer,
        ) -> Cow<'static, str> {
            unreachable!("`as_string_threadsafe` is not used in non-parallel rustc")
        }

        #[cfg(parallel_compiler)]
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
            let baked_data_provider = rustc_baked_icu_data::baked_data_provider();
            let locale_fallbacker =
                LocaleFallbacker::try_new_with_any_provider(&baked_data_provider)
                    .expect("Failed to create fallback provider");
            let data_provider =
                LocaleFallbackProvider::new_with_fallbacker(baked_data_provider, locale_fallbacker);
            let locale = icu_locale_from_unic_langid(lang)
                .unwrap_or_else(|| rustc_baked_icu_data::supported_locales::EN);
            let list_formatter =
                icu_list::ListFormatter::try_new_and_with_length_with_any_provider(
                    &data_provider,
                    &locale.into(),
                    icu_list::ListLength::Wide,
                )
                .expect("Failed to create list formatter");

            Ok(MemoizableListFormatter(list_formatter))
        }
    }

    let l = l.into_iter().map(|x| x.into_owned()).collect();

    FluentValue::Custom(Box::new(FluentStrListSepByAnd(l)))
}
