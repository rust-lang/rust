use std::borrow::Cow;
use std::fmt::{self, Debug};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::panic;
use std::path::PathBuf;
use std::thread::panicking;

use rustc_data_structures::fx::FxIndexMap;
use rustc_error_messages::{FluentValue, fluent_value_from_str_list_sep_by_and};
use rustc_lint_defs::{Applicability, LintExpectationId};
use rustc_macros::{Decodable, Encodable};
use rustc_span::source_map::Spanned;
use rustc_span::{DUMMY_SP, Span, Symbol};
use tracing::debug;

use crate::snippet::Style;
use crate::{
    CodeSuggestion, DiagCtxtHandle, DiagMessage, ErrCode, ErrorGuaranteed, ExplicitBug, Level,
    MultiSpan, StashKey, SubdiagMessage, Substitution, SubstitutionPart, SuggestionStyle,
    Suggestions,
};

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

pub type DiagArgMap = FxIndexMap<DiagArgName, DiagArgValue>;

/// Trait for types that `Diag::emit` can return as a "guarantee" (or "proof")
/// token that the emission happened.
pub trait EmissionGuarantee: Sized {
    /// This exists so that bugs and fatal errors can both result in `!` (an
    /// abort) when emitted, but have different aborting behaviour.
    type EmitResult = Self;

    /// Implementation of `Diag::emit`, fully controlled by each `impl` of
    /// `EmissionGuarantee`, to make it impossible to create a value of
    /// `Self::EmitResult` without actually performing the emission.
    #[track_caller]
    fn emit_producing_guarantee(diag: Diag<'_, Self>) -> Self::EmitResult;
}

impl EmissionGuarantee for ErrorGuaranteed {
    fn emit_producing_guarantee(diag: Diag<'_, Self>) -> Self::EmitResult {
        diag.emit_producing_error_guaranteed()
    }
}

impl EmissionGuarantee for () {
    fn emit_producing_guarantee(diag: Diag<'_, Self>) -> Self::EmitResult {
        diag.emit_producing_nothing();
    }
}

/// Marker type which enables implementation of `create_bug` and `emit_bug` functions for
/// bug diagnostics.
#[derive(Copy, Clone)]
pub struct BugAbort;

impl EmissionGuarantee for BugAbort {
    type EmitResult = !;

    fn emit_producing_guarantee(diag: Diag<'_, Self>) -> Self::EmitResult {
        diag.emit_producing_nothing();
        panic::panic_any(ExplicitBug);
    }
}

/// Marker type which enables implementation of `create_fatal` and `emit_fatal` functions for
/// fatal diagnostics.
#[derive(Copy, Clone)]
pub struct FatalAbort;

impl EmissionGuarantee for FatalAbort {
    type EmitResult = !;

    fn emit_producing_guarantee(diag: Diag<'_, Self>) -> Self::EmitResult {
        diag.emit_producing_nothing();
        crate::FatalError.raise()
    }
}

impl EmissionGuarantee for rustc_span::fatal_error::FatalError {
    fn emit_producing_guarantee(diag: Diag<'_, Self>) -> Self::EmitResult {
        diag.emit_producing_nothing();
        rustc_span::fatal_error::FatalError
    }
}

/// Trait implemented by error types. This is rarely implemented manually. Instead, use
/// `#[derive(Diagnostic)]` -- see [rustc_macros::Diagnostic].
///
/// When implemented manually, it should be generic over the emission
/// guarantee, i.e.:
/// ```ignore (fragment)
/// impl<'a, G: EmissionGuarantee> Diagnostic<'a, G> for Foo { ... }
/// ```
/// rather than being specific:
/// ```ignore (fragment)
/// impl<'a> Diagnostic<'a> for Bar { ... }  // the default type param is `ErrorGuaranteed`
/// impl<'a> Diagnostic<'a, ()> for Baz { ... }
/// ```
/// There are two reasons for this.
/// - A diagnostic like `Foo` *could* be emitted at any level -- `level` is
///   passed in to `into_diag` from outside. Even if in practice it is
///   always emitted at a single level, we let the diagnostic creation/emission
///   site determine the level (by using `create_err`, `emit_warn`, etc.)
///   rather than the `Diagnostic` impl.
/// - Derived impls are always generic, and it's good for the hand-written
///   impls to be consistent with them.
#[rustc_diagnostic_item = "Diagnostic"]
pub trait Diagnostic<'a, G: EmissionGuarantee = ErrorGuaranteed> {
    /// Write out as a diagnostic out of `DiagCtxt`.
    #[must_use]
    fn into_diag(self, dcx: DiagCtxtHandle<'a>, level: Level) -> Diag<'a, G>;
}

impl<'a, T, G> Diagnostic<'a, G> for Spanned<T>
where
    T: Diagnostic<'a, G>,
    G: EmissionGuarantee,
{
    fn into_diag(self, dcx: DiagCtxtHandle<'a>, level: Level) -> Diag<'a, G> {
        self.node.into_diag(dcx, level).with_span(self.span)
    }
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

/// Trait implemented by error types. This should not be implemented manually. Instead, use
/// `#[derive(Subdiagnostic)]` -- see [rustc_macros::Subdiagnostic].
#[rustc_diagnostic_item = "Subdiagnostic"]
pub trait Subdiagnostic
where
    Self: Sized,
{
    /// Add a subdiagnostic to an existing diagnostic.
    fn add_to_diag<G: EmissionGuarantee>(self, diag: &mut Diag<'_, G>);
}

/// Trait implemented by lint types. This should not be implemented manually. Instead, use
/// `#[derive(LintDiagnostic)]` -- see [rustc_macros::LintDiagnostic].
#[rustc_diagnostic_item = "LintDiagnostic"]
pub trait LintDiagnostic<'a, G: EmissionGuarantee> {
    /// Decorate and emit a lint.
    fn decorate_lint<'b>(self, diag: &'b mut Diag<'a, G>);
}

#[derive(Clone, Debug, Encodable, Decodable)]
pub(crate) struct DiagLocation {
    file: Cow<'static, str>,
    line: u32,
    col: u32,
}

impl DiagLocation {
    #[track_caller]
    fn caller() -> Self {
        let loc = panic::Location::caller();
        DiagLocation { file: loc.file().into(), line: loc.line(), col: loc.column() }
    }
}

impl fmt::Display for DiagLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.file, self.line, self.col)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Encodable, Decodable)]
pub struct IsLint {
    /// The lint name.
    pub(crate) name: String,
    /// Indicates whether this lint should show up in cargo's future breakage report.
    has_future_breakage: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub struct DiagStyledString(pub Vec<StringPart>);

impl DiagStyledString {
    pub fn new() -> DiagStyledString {
        DiagStyledString(vec![])
    }
    pub fn push_normal<S: Into<String>>(&mut self, t: S) {
        self.0.push(StringPart::normal(t));
    }
    pub fn push_highlighted<S: Into<String>>(&mut self, t: S) {
        self.0.push(StringPart::highlighted(t));
    }
    pub fn push<S: Into<String>>(&mut self, t: S, highlight: bool) {
        if highlight {
            self.push_highlighted(t);
        } else {
            self.push_normal(t);
        }
    }
    pub fn normal<S: Into<String>>(t: S) -> DiagStyledString {
        DiagStyledString(vec![StringPart::normal(t)])
    }

    pub fn highlighted<S: Into<String>>(t: S) -> DiagStyledString {
        DiagStyledString(vec![StringPart::highlighted(t)])
    }

    pub fn content(&self) -> String {
        self.0.iter().map(|x| x.content.as_str()).collect::<String>()
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct StringPart {
    content: String,
    style: Style,
}

impl StringPart {
    pub fn normal<S: Into<String>>(content: S) -> StringPart {
        StringPart { content: content.into(), style: Style::NoStyle }
    }

    pub fn highlighted<S: Into<String>>(content: S) -> StringPart {
        StringPart { content: content.into(), style: Style::Highlight }
    }
}

/// The main part of a diagnostic. Note that `Diag`, which wraps this type, is
/// used for most operations, and should be used instead whenever possible.
/// This type should only be used when `Diag`'s lifetime causes difficulties,
/// e.g. when storing diagnostics within `DiagCtxt`.
#[must_use]
#[derive(Clone, Debug, Encodable, Decodable)]
pub struct DiagInner {
    // NOTE(eddyb) this is private to disallow arbitrary after-the-fact changes,
    // outside of what methods in this crate themselves allow.
    pub(crate) level: Level,

    pub messages: Vec<(DiagMessage, Style)>,
    pub code: Option<ErrCode>,
    pub lint_id: Option<LintExpectationId>,
    pub span: MultiSpan,
    pub children: Vec<Subdiag>,
    pub suggestions: Suggestions,
    pub args: DiagArgMap,

    // This is used to store args and restore them after a subdiagnostic is rendered.
    pub reserved_args: DiagArgMap,

    /// This is not used for highlighting or rendering any error message. Rather, it can be used
    /// as a sort key to sort a buffer of diagnostics. By default, it is the primary span of
    /// `span` if there is one. Otherwise, it is `DUMMY_SP`.
    pub sort_span: Span,

    pub is_lint: Option<IsLint>,

    pub long_ty_path: Option<PathBuf>,
    /// With `-Ztrack_diagnostics` enabled,
    /// we print where in rustc this error was emitted.
    pub(crate) emitted_at: DiagLocation,
}

impl DiagInner {
    #[track_caller]
    pub fn new<M: Into<DiagMessage>>(level: Level, message: M) -> Self {
        DiagInner::new_with_messages(level, vec![(message.into(), Style::NoStyle)])
    }

    #[track_caller]
    pub fn new_with_messages(level: Level, messages: Vec<(DiagMessage, Style)>) -> Self {
        DiagInner {
            level,
            lint_id: None,
            messages,
            code: None,
            span: MultiSpan::new(),
            children: vec![],
            suggestions: Suggestions::Enabled(vec![]),
            args: Default::default(),
            reserved_args: Default::default(),
            sort_span: DUMMY_SP,
            is_lint: None,
            long_ty_path: None,
            emitted_at: DiagLocation::caller(),
        }
    }

    #[inline(always)]
    pub fn level(&self) -> Level {
        self.level
    }

    pub fn is_error(&self) -> bool {
        match self.level {
            Level::Bug | Level::Fatal | Level::Error | Level::DelayedBug => true,

            Level::ForceWarning
            | Level::Warning
            | Level::Note
            | Level::OnceNote
            | Level::Help
            | Level::OnceHelp
            | Level::FailureNote
            | Level::Allow
            | Level::Expect => false,
        }
    }

    /// Indicates whether this diagnostic should show up in cargo's future breakage report.
    pub(crate) fn has_future_breakage(&self) -> bool {
        matches!(self.is_lint, Some(IsLint { has_future_breakage: true, .. }))
    }

    pub(crate) fn is_force_warn(&self) -> bool {
        match self.level {
            Level::ForceWarning => {
                assert!(self.is_lint.is_some());
                true
            }
            _ => false,
        }
    }

    // See comment on `Diag::subdiagnostic_message_to_diagnostic_message`.
    pub(crate) fn subdiagnostic_message_to_diagnostic_message(
        &self,
        attr: impl Into<SubdiagMessage>,
    ) -> DiagMessage {
        let msg =
            self.messages.iter().map(|(msg, _)| msg).next().expect("diagnostic with no messages");
        msg.with_subdiagnostic_message(attr.into())
    }

    pub(crate) fn sub(
        &mut self,
        level: Level,
        message: impl Into<SubdiagMessage>,
        span: MultiSpan,
    ) {
        let sub = Subdiag {
            level,
            messages: vec![(
                self.subdiagnostic_message_to_diagnostic_message(message),
                Style::NoStyle,
            )],
            span,
        };
        self.children.push(sub);
    }

    pub(crate) fn arg(&mut self, name: impl Into<DiagArgName>, arg: impl IntoDiagArg) {
        let name = name.into();
        let value = arg.into_diag_arg(&mut self.long_ty_path);
        // This assertion is to avoid subdiagnostics overwriting an existing diagnostic arg.
        debug_assert!(
            !self.args.contains_key(&name) || self.args.get(&name) == Some(&value),
            "arg {} already exists",
            name
        );
        self.args.insert(name, value);
    }

    pub fn remove_arg(&mut self, name: &str) {
        self.args.swap_remove(name);
    }

    pub fn store_args(&mut self) {
        self.reserved_args = self.args.clone();
    }

    pub fn restore_args(&mut self) {
        self.args = std::mem::take(&mut self.reserved_args);
    }

    /// Fields used for Hash, and PartialEq trait.
    fn keys(
        &self,
    ) -> (
        &Level,
        &[(DiagMessage, Style)],
        &Option<ErrCode>,
        &MultiSpan,
        &[Subdiag],
        &Suggestions,
        Vec<(&DiagArgName, &DiagArgValue)>,
        &Option<IsLint>,
    ) {
        (
            &self.level,
            &self.messages,
            &self.code,
            &self.span,
            &self.children,
            &self.suggestions,
            self.args.iter().collect(),
            // omit self.sort_span
            &self.is_lint,
            // omit self.emitted_at
        )
    }
}

impl Hash for DiagInner {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.keys().hash(state);
    }
}

impl PartialEq for DiagInner {
    fn eq(&self, other: &Self) -> bool {
        self.keys() == other.keys()
    }
}

/// A "sub"-diagnostic attached to a parent diagnostic.
/// For example, a note attached to an error.
#[derive(Clone, Debug, PartialEq, Hash, Encodable, Decodable)]
pub struct Subdiag {
    pub level: Level,
    pub messages: Vec<(DiagMessage, Style)>,
    pub span: MultiSpan,
}

/// Used for emitting structured error messages and other diagnostic information.
/// Wraps a `DiagInner`, adding some useful things.
/// - The `dcx` field, allowing it to (a) emit itself, and (b) do a drop check
///   that it has been emitted or cancelled.
/// - The `EmissionGuarantee`, which determines the type returned from `emit`.
///
/// Each constructed `Diag` must be consumed by a function such as `emit`,
/// `cancel`, `delay_as_bug`, or `into_diag`. A panic occurs if a `Diag`
/// is dropped without being consumed by one of these functions.
///
/// If there is some state in a downstream crate you would like to access in
/// the methods of `Diag` here, consider extending `DiagCtxtFlags`.
#[must_use]
pub struct Diag<'a, G: EmissionGuarantee = ErrorGuaranteed> {
    pub dcx: DiagCtxtHandle<'a>,

    /// Why the `Option`? It is always `Some` until the `Diag` is consumed via
    /// `emit`, `cancel`, etc. At that point it is consumed and replaced with
    /// `None`. Then `drop` checks that it is `None`; if not, it panics because
    /// a diagnostic was built but not used.
    ///
    /// Why the Box? `DiagInner` is a large type, and `Diag` is often used as a
    /// return value, especially within the frequently-used `PResult` type. In
    /// theory, return value optimization (RVO) should avoid unnecessary
    /// copying. In practice, it does not (at the time of writing).
    diag: Option<Box<DiagInner>>,

    _marker: PhantomData<G>,
}

// Cloning a `Diag` is a recipe for a diagnostic being emitted twice, which
// would be bad.
impl<G> !Clone for Diag<'_, G> {}

rustc_data_structures::static_assert_size!(Diag<'_, ()>, 3 * size_of::<usize>());

impl<G: EmissionGuarantee> Deref for Diag<'_, G> {
    type Target = DiagInner;

    fn deref(&self) -> &DiagInner {
        self.diag.as_ref().unwrap()
    }
}

impl<G: EmissionGuarantee> DerefMut for Diag<'_, G> {
    fn deref_mut(&mut self) -> &mut DiagInner {
        self.diag.as_mut().unwrap()
    }
}

impl<G: EmissionGuarantee> Debug for Diag<'_, G> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.diag.fmt(f)
    }
}

/// `Diag` impls many `&mut self -> &mut Self` methods. Each one modifies an
/// existing diagnostic, either in a standalone fashion, e.g.
/// `err.code(code);`, or in a chained fashion to make multiple modifications,
/// e.g. `err.code(code).span(span);`.
///
/// This macro creates an equivalent `self -> Self` method, with a `with_`
/// prefix. This can be used in a chained fashion when making a new diagnostic,
/// e.g. `let err = struct_err(msg).with_code(code);`, or emitting a new
/// diagnostic, e.g. `struct_err(msg).with_code(code).emit();`.
///
/// Although the latter method can be used to modify an existing diagnostic,
/// e.g. `err = err.with_code(code);`, this should be avoided because the former
/// method gives shorter code, e.g. `err.code(code);`.
///
/// Note: the `with_` methods are added only when needed. If you want to use
/// one and it's not defined, feel free to add it.
///
/// Note: any doc comments must be within the `with_fn!` call.
macro_rules! with_fn {
    {
        $with_f:ident,
        $(#[$attrs:meta])*
        pub fn $f:ident(&mut $self:ident, $($name:ident: $ty:ty),* $(,)?) -> &mut Self {
            $($body:tt)*
        }
    } => {
        // The original function.
        $(#[$attrs])*
        #[doc = concat!("See [`Diag::", stringify!($f), "()`].")]
        pub fn $f(&mut $self, $($name: $ty),*) -> &mut Self {
            $($body)*
        }

        // The `with_*` variant.
        $(#[$attrs])*
        #[doc = concat!("See [`Diag::", stringify!($f), "()`].")]
        pub fn $with_f(mut $self, $($name: $ty),*) -> Self {
            $self.$f($($name),*);
            $self
        }
    };
}

impl<'a, G: EmissionGuarantee> Diag<'a, G> {
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn new(dcx: DiagCtxtHandle<'a>, level: Level, message: impl Into<DiagMessage>) -> Self {
        Self::new_diagnostic(dcx, DiagInner::new(level, message))
    }

    /// Allow moving diagnostics between different error tainting contexts
    pub fn with_dcx(mut self, dcx: DiagCtxtHandle<'_>) -> Diag<'_, G> {
        Diag { dcx, diag: self.diag.take(), _marker: PhantomData }
    }

    /// Creates a new `Diag` with an already constructed diagnostic.
    #[track_caller]
    pub(crate) fn new_diagnostic(dcx: DiagCtxtHandle<'a>, diag: DiagInner) -> Self {
        debug!("Created new diagnostic");
        Self { dcx, diag: Some(Box::new(diag)), _marker: PhantomData }
    }

    /// Delay emission of this diagnostic as a bug.
    ///
    /// This can be useful in contexts where an error indicates a bug but
    /// typically this only happens when other compilation errors have already
    /// happened. In those cases this can be used to defer emission of this
    /// diagnostic as a bug in the compiler only if no other errors have been
    /// emitted.
    ///
    /// In the meantime, though, callsites are required to deal with the "bug"
    /// locally in whichever way makes the most sense.
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn downgrade_to_delayed_bug(&mut self) {
        assert!(
            matches!(self.level, Level::Error | Level::DelayedBug),
            "downgrade_to_delayed_bug: cannot downgrade {:?} to DelayedBug: not an error",
            self.level
        );
        self.level = Level::DelayedBug;
    }

    with_fn! { with_span_label,
    /// Appends a labeled span to the diagnostic.
    ///
    /// Labels are used to convey additional context for the diagnostic's primary span. They will
    /// be shown together with the original diagnostic's span, *not* with spans added by
    /// `span_note`, `span_help`, etc. Therefore, if the primary span is not displayable (because
    /// the span is `DUMMY_SP` or the source code isn't found), labels will not be displayed
    /// either.
    ///
    /// Implementation-wise, the label span is pushed onto the [`MultiSpan`] that was created when
    /// the diagnostic was constructed. However, the label span is *not* considered a
    /// ["primary span"][`MultiSpan`]; only the `Span` supplied when creating the diagnostic is
    /// primary.
    #[rustc_lint_diagnostics]
    pub fn span_label(&mut self, span: Span, label: impl Into<SubdiagMessage>) -> &mut Self {
        let msg = self.subdiagnostic_message_to_diagnostic_message(label);
        self.span.push_span_label(span, msg);
        self
    } }

    with_fn! { with_span_labels,
    /// Labels all the given spans with the provided label.
    /// See [`Self::span_label()`] for more information.
    #[rustc_lint_diagnostics]
    pub fn span_labels(&mut self, spans: impl IntoIterator<Item = Span>, label: &str) -> &mut Self {
        for span in spans {
            self.span_label(span, label.to_string());
        }
        self
    } }

    #[rustc_lint_diagnostics]
    pub fn replace_span_with(&mut self, after: Span, keep_label: bool) -> &mut Self {
        let before = self.span.clone();
        self.span(after);
        for span_label in before.span_labels() {
            if let Some(label) = span_label.label {
                if span_label.is_primary && keep_label {
                    self.span.push_span_label(after, label);
                } else {
                    self.span.push_span_label(span_label.span, label);
                }
            }
        }
        self
    }

    #[rustc_lint_diagnostics]
    pub fn note_expected_found(
        &mut self,
        expected_label: &str,
        expected: DiagStyledString,
        found_label: &str,
        found: DiagStyledString,
    ) -> &mut Self {
        self.note_expected_found_extra(
            expected_label,
            expected,
            found_label,
            found,
            DiagStyledString::normal(""),
            DiagStyledString::normal(""),
        )
    }

    #[rustc_lint_diagnostics]
    pub fn note_expected_found_extra(
        &mut self,
        expected_label: &str,
        expected: DiagStyledString,
        found_label: &str,
        found: DiagStyledString,
        expected_extra: DiagStyledString,
        found_extra: DiagStyledString,
    ) -> &mut Self {
        let expected_label = expected_label.to_string();
        let expected_label = if expected_label.is_empty() {
            "expected".to_string()
        } else {
            format!("expected {expected_label}")
        };
        let found_label = found_label.to_string();
        let found_label = if found_label.is_empty() {
            "found".to_string()
        } else {
            format!("found {found_label}")
        };
        let (found_padding, expected_padding) = if expected_label.len() > found_label.len() {
            (expected_label.len() - found_label.len(), 0)
        } else {
            (0, found_label.len() - expected_label.len())
        };
        let mut msg = vec![StringPart::normal(format!(
            "{}{} `",
            " ".repeat(expected_padding),
            expected_label
        ))];
        msg.extend(expected.0);
        msg.push(StringPart::normal(format!("`")));
        msg.extend(expected_extra.0);
        msg.push(StringPart::normal(format!("\n")));
        msg.push(StringPart::normal(format!("{}{} `", " ".repeat(found_padding), found_label)));
        msg.extend(found.0);
        msg.push(StringPart::normal(format!("`")));
        msg.extend(found_extra.0);

        // For now, just attach these as notes.
        self.highlighted_note(msg);
        self
    }

    #[rustc_lint_diagnostics]
    pub fn note_trait_signature(&mut self, name: Symbol, signature: String) -> &mut Self {
        self.highlighted_note(vec![
            StringPart::normal(format!("`{name}` from trait: `")),
            StringPart::highlighted(signature),
            StringPart::normal("`"),
        ]);
        self
    }

    with_fn! { with_note,
    /// Add a note attached to this diagnostic.
    #[rustc_lint_diagnostics]
    pub fn note(&mut self, msg: impl Into<SubdiagMessage>) -> &mut Self {
        self.sub(Level::Note, msg, MultiSpan::new());
        self
    } }

    #[rustc_lint_diagnostics]
    pub fn highlighted_note(&mut self, msg: Vec<StringPart>) -> &mut Self {
        self.sub_with_highlights(Level::Note, msg, MultiSpan::new());
        self
    }

    #[rustc_lint_diagnostics]
    pub fn highlighted_span_note(
        &mut self,
        span: impl Into<MultiSpan>,
        msg: Vec<StringPart>,
    ) -> &mut Self {
        self.sub_with_highlights(Level::Note, msg, span.into());
        self
    }

    /// This is like [`Diag::note()`], but it's only printed once.
    #[rustc_lint_diagnostics]
    pub fn note_once(&mut self, msg: impl Into<SubdiagMessage>) -> &mut Self {
        self.sub(Level::OnceNote, msg, MultiSpan::new());
        self
    }

    with_fn! { with_span_note,
    /// Prints the span with a note above it.
    /// This is like [`Diag::note()`], but it gets its own span.
    #[rustc_lint_diagnostics]
    pub fn span_note(
        &mut self,
        sp: impl Into<MultiSpan>,
        msg: impl Into<SubdiagMessage>,
    ) -> &mut Self {
        self.sub(Level::Note, msg, sp.into());
        self
    } }

    /// Prints the span with a note above it.
    /// This is like [`Diag::note_once()`], but it gets its own span.
    #[rustc_lint_diagnostics]
    pub fn span_note_once<S: Into<MultiSpan>>(
        &mut self,
        sp: S,
        msg: impl Into<SubdiagMessage>,
    ) -> &mut Self {
        self.sub(Level::OnceNote, msg, sp.into());
        self
    }

    with_fn! { with_warn,
    /// Add a warning attached to this diagnostic.
    #[rustc_lint_diagnostics]
    pub fn warn(&mut self, msg: impl Into<SubdiagMessage>) -> &mut Self {
        self.sub(Level::Warning, msg, MultiSpan::new());
        self
    } }

    /// Prints the span with a warning above it.
    /// This is like [`Diag::warn()`], but it gets its own span.
    #[rustc_lint_diagnostics]
    pub fn span_warn<S: Into<MultiSpan>>(
        &mut self,
        sp: S,
        msg: impl Into<SubdiagMessage>,
    ) -> &mut Self {
        self.sub(Level::Warning, msg, sp.into());
        self
    }

    with_fn! { with_help,
    /// Add a help message attached to this diagnostic.
    #[rustc_lint_diagnostics]
    pub fn help(&mut self, msg: impl Into<SubdiagMessage>) -> &mut Self {
        self.sub(Level::Help, msg, MultiSpan::new());
        self
    } }

    /// This is like [`Diag::help()`], but it's only printed once.
    #[rustc_lint_diagnostics]
    pub fn help_once(&mut self, msg: impl Into<SubdiagMessage>) -> &mut Self {
        self.sub(Level::OnceHelp, msg, MultiSpan::new());
        self
    }

    /// Add a help message attached to this diagnostic with a customizable highlighted message.
    #[rustc_lint_diagnostics]
    pub fn highlighted_help(&mut self, msg: Vec<StringPart>) -> &mut Self {
        self.sub_with_highlights(Level::Help, msg, MultiSpan::new());
        self
    }

    /// Add a help message attached to this diagnostic with a customizable highlighted message.
    #[rustc_lint_diagnostics]
    pub fn highlighted_span_help(
        &mut self,
        span: impl Into<MultiSpan>,
        msg: Vec<StringPart>,
    ) -> &mut Self {
        self.sub_with_highlights(Level::Help, msg, span.into());
        self
    }

    /// Prints the span with some help above it.
    /// This is like [`Diag::help()`], but it gets its own span.
    #[rustc_lint_diagnostics]
    pub fn span_help<S: Into<MultiSpan>>(
        &mut self,
        sp: S,
        msg: impl Into<SubdiagMessage>,
    ) -> &mut Self {
        self.sub(Level::Help, msg, sp.into());
        self
    }

    /// Disallow attaching suggestions to this diagnostic.
    /// Any suggestions attached e.g. with the `span_suggestion_*` methods
    /// (before and after the call to `disable_suggestions`) will be ignored.
    #[rustc_lint_diagnostics]
    pub fn disable_suggestions(&mut self) -> &mut Self {
        self.suggestions = Suggestions::Disabled;
        self
    }

    /// Prevent new suggestions from being added to this diagnostic.
    ///
    /// Suggestions added before the call to `.seal_suggestions()` will be preserved
    /// and new suggestions will be ignored.
    #[rustc_lint_diagnostics]
    pub fn seal_suggestions(&mut self) -> &mut Self {
        if let Suggestions::Enabled(suggestions) = &mut self.suggestions {
            let suggestions_slice = std::mem::take(suggestions).into_boxed_slice();
            self.suggestions = Suggestions::Sealed(suggestions_slice);
        }
        self
    }

    /// Helper for pushing to `self.suggestions`.
    ///
    /// A new suggestion is added if suggestions are enabled for this diagnostic.
    /// Otherwise, they are ignored.
    #[rustc_lint_diagnostics]
    fn push_suggestion(&mut self, suggestion: CodeSuggestion) {
        for subst in &suggestion.substitutions {
            for part in &subst.parts {
                let span = part.span;
                let call_site = span.ctxt().outer_expn_data().call_site;
                if span.in_derive_expansion() && span.overlaps_or_adjacent(call_site) {
                    // Ignore if spans is from derive macro.
                    return;
                }
            }
        }

        if let Suggestions::Enabled(suggestions) = &mut self.suggestions {
            suggestions.push(suggestion);
        }
    }

    with_fn! { with_multipart_suggestion,
    /// Show a suggestion that has multiple parts to it.
    /// In other words, multiple changes need to be applied as part of this suggestion.
    #[rustc_lint_diagnostics]
    pub fn multipart_suggestion(
        &mut self,
        msg: impl Into<SubdiagMessage>,
        suggestion: Vec<(Span, String)>,
        applicability: Applicability,
    ) -> &mut Self {
        self.multipart_suggestion_with_style(
            msg,
            suggestion,
            applicability,
            SuggestionStyle::ShowCode,
        )
    } }

    /// Show a suggestion that has multiple parts to it, always as its own subdiagnostic.
    /// In other words, multiple changes need to be applied as part of this suggestion.
    #[rustc_lint_diagnostics]
    pub fn multipart_suggestion_verbose(
        &mut self,
        msg: impl Into<SubdiagMessage>,
        suggestion: Vec<(Span, String)>,
        applicability: Applicability,
    ) -> &mut Self {
        self.multipart_suggestion_with_style(
            msg,
            suggestion,
            applicability,
            SuggestionStyle::ShowAlways,
        )
    }

    /// [`Diag::multipart_suggestion()`] but you can set the [`SuggestionStyle`].
    #[rustc_lint_diagnostics]
    pub fn multipart_suggestion_with_style(
        &mut self,
        msg: impl Into<SubdiagMessage>,
        mut suggestion: Vec<(Span, String)>,
        applicability: Applicability,
        style: SuggestionStyle,
    ) -> &mut Self {
        let mut seen = crate::FxHashSet::default();
        suggestion.retain(|(span, msg)| seen.insert((span.lo(), span.hi(), msg.clone())));

        let parts = suggestion
            .into_iter()
            .map(|(span, snippet)| SubstitutionPart { snippet, span })
            .collect::<Vec<_>>();

        assert!(!parts.is_empty());
        debug_assert_eq!(
            parts.iter().find(|part| part.span.is_empty() && part.snippet.is_empty()),
            None,
            "Span must not be empty and have no suggestion",
        );
        debug_assert_eq!(
            parts.array_windows().find(|[a, b]| a.span.overlaps(b.span)),
            None,
            "suggestion must not have overlapping parts",
        );

        self.push_suggestion(CodeSuggestion {
            substitutions: vec![Substitution { parts }],
            msg: self.subdiagnostic_message_to_diagnostic_message(msg),
            style,
            applicability,
        });
        self
    }

    /// Prints out a message with for a multipart suggestion without showing the suggested code.
    ///
    /// This is intended to be used for suggestions that are obvious in what the changes need to
    /// be from the message, showing the span label inline would be visually unpleasant
    /// (marginally overlapping spans or multiline spans) and showing the snippet window wouldn't
    /// improve understandability.
    #[rustc_lint_diagnostics]
    pub fn tool_only_multipart_suggestion(
        &mut self,
        msg: impl Into<SubdiagMessage>,
        suggestion: Vec<(Span, String)>,
        applicability: Applicability,
    ) -> &mut Self {
        self.multipart_suggestion_with_style(
            msg,
            suggestion,
            applicability,
            SuggestionStyle::CompletelyHidden,
        )
    }

    with_fn! { with_span_suggestion,
    /// Prints out a message with a suggested edit of the code.
    ///
    /// In case of short messages and a simple suggestion, rustc displays it as a label:
    ///
    /// ```text
    /// try adding parentheses: `(tup.0).1`
    /// ```
    ///
    /// The message
    ///
    /// * should not end in any punctuation (a `:` is added automatically)
    /// * should not be a question (avoid language like "did you mean")
    /// * should not contain any phrases like "the following", "as shown", etc.
    /// * may look like "to do xyz, use" or "to do xyz, use abc"
    /// * may contain a name of a function, variable, or type, but not whole expressions
    ///
    /// See [`CodeSuggestion`] for more information.
    #[rustc_lint_diagnostics]
    pub fn span_suggestion(
        &mut self,
        sp: Span,
        msg: impl Into<SubdiagMessage>,
        suggestion: impl ToString,
        applicability: Applicability,
    ) -> &mut Self {
        self.span_suggestion_with_style(
            sp,
            msg,
            suggestion,
            applicability,
            SuggestionStyle::ShowCode,
        );
        self
    } }

    /// [`Diag::span_suggestion()`] but you can set the [`SuggestionStyle`].
    #[rustc_lint_diagnostics]
    pub fn span_suggestion_with_style(
        &mut self,
        sp: Span,
        msg: impl Into<SubdiagMessage>,
        suggestion: impl ToString,
        applicability: Applicability,
        style: SuggestionStyle,
    ) -> &mut Self {
        debug_assert!(
            !(sp.is_empty() && suggestion.to_string().is_empty()),
            "Span must not be empty and have no suggestion"
        );
        self.push_suggestion(CodeSuggestion {
            substitutions: vec![Substitution {
                parts: vec![SubstitutionPart { snippet: suggestion.to_string(), span: sp }],
            }],
            msg: self.subdiagnostic_message_to_diagnostic_message(msg),
            style,
            applicability,
        });
        self
    }

    with_fn! { with_span_suggestion_verbose,
    /// Always show the suggested change.
    #[rustc_lint_diagnostics]
    pub fn span_suggestion_verbose(
        &mut self,
        sp: Span,
        msg: impl Into<SubdiagMessage>,
        suggestion: impl ToString,
        applicability: Applicability,
    ) -> &mut Self {
        self.span_suggestion_with_style(
            sp,
            msg,
            suggestion,
            applicability,
            SuggestionStyle::ShowAlways,
        );
        self
    } }

    with_fn! { with_span_suggestions,
    /// Prints out a message with multiple suggested edits of the code.
    /// See also [`Diag::span_suggestion()`].
    #[rustc_lint_diagnostics]
    pub fn span_suggestions(
        &mut self,
        sp: Span,
        msg: impl Into<SubdiagMessage>,
        suggestions: impl IntoIterator<Item = String>,
        applicability: Applicability,
    ) -> &mut Self {
        self.span_suggestions_with_style(
            sp,
            msg,
            suggestions,
            applicability,
            SuggestionStyle::ShowCode,
        )
    } }

    #[rustc_lint_diagnostics]
    pub fn span_suggestions_with_style(
        &mut self,
        sp: Span,
        msg: impl Into<SubdiagMessage>,
        suggestions: impl IntoIterator<Item = String>,
        applicability: Applicability,
        style: SuggestionStyle,
    ) -> &mut Self {
        let substitutions = suggestions
            .into_iter()
            .map(|snippet| {
                debug_assert!(
                    !(sp.is_empty() && snippet.is_empty()),
                    "Span must not be empty and have no suggestion"
                );
                Substitution { parts: vec![SubstitutionPart { snippet, span: sp }] }
            })
            .collect();
        self.push_suggestion(CodeSuggestion {
            substitutions,
            msg: self.subdiagnostic_message_to_diagnostic_message(msg),
            style,
            applicability,
        });
        self
    }

    /// Prints out a message with multiple suggested edits of the code, where each edit consists of
    /// multiple parts.
    /// See also [`Diag::multipart_suggestion()`].
    #[rustc_lint_diagnostics]
    pub fn multipart_suggestions(
        &mut self,
        msg: impl Into<SubdiagMessage>,
        suggestions: impl IntoIterator<Item = Vec<(Span, String)>>,
        applicability: Applicability,
    ) -> &mut Self {
        let substitutions = suggestions
            .into_iter()
            .map(|sugg| {
                let mut parts = sugg
                    .into_iter()
                    .map(|(span, snippet)| SubstitutionPart { snippet, span })
                    .collect::<Vec<_>>();

                parts.sort_unstable_by_key(|part| part.span);

                assert!(!parts.is_empty());
                debug_assert_eq!(
                    parts.iter().find(|part| part.span.is_empty() && part.snippet.is_empty()),
                    None,
                    "Span must not be empty and have no suggestion",
                );
                debug_assert_eq!(
                    parts.array_windows().find(|[a, b]| a.span.overlaps(b.span)),
                    None,
                    "suggestion must not have overlapping parts",
                );

                Substitution { parts }
            })
            .collect();

        self.push_suggestion(CodeSuggestion {
            substitutions,
            msg: self.subdiagnostic_message_to_diagnostic_message(msg),
            style: SuggestionStyle::ShowCode,
            applicability,
        });
        self
    }

    with_fn! { with_span_suggestion_short,
    /// Prints out a message with a suggested edit of the code. If the suggestion is presented
    /// inline, it will only show the message and not the suggestion.
    ///
    /// See [`CodeSuggestion`] for more information.
    #[rustc_lint_diagnostics]
    pub fn span_suggestion_short(
        &mut self,
        sp: Span,
        msg: impl Into<SubdiagMessage>,
        suggestion: impl ToString,
        applicability: Applicability,
    ) -> &mut Self {
        self.span_suggestion_with_style(
            sp,
            msg,
            suggestion,
            applicability,
            SuggestionStyle::HideCodeInline,
        );
        self
    } }

    /// Prints out a message for a suggestion without showing the suggested code.
    ///
    /// This is intended to be used for suggestions that are obvious in what the changes need to
    /// be from the message, showing the span label inline would be visually unpleasant
    /// (marginally overlapping spans or multiline spans) and showing the snippet window wouldn't
    /// improve understandability.
    #[rustc_lint_diagnostics]
    pub fn span_suggestion_hidden(
        &mut self,
        sp: Span,
        msg: impl Into<SubdiagMessage>,
        suggestion: impl ToString,
        applicability: Applicability,
    ) -> &mut Self {
        self.span_suggestion_with_style(
            sp,
            msg,
            suggestion,
            applicability,
            SuggestionStyle::HideCodeAlways,
        );
        self
    }

    with_fn! { with_tool_only_span_suggestion,
    /// Adds a suggestion to the JSON output that will not be shown in the CLI.
    ///
    /// This is intended to be used for suggestions that are *very* obvious in what the changes
    /// need to be from the message, but we still want other tools to be able to apply them.
    #[rustc_lint_diagnostics]
    pub fn tool_only_span_suggestion(
        &mut self,
        sp: Span,
        msg: impl Into<SubdiagMessage>,
        suggestion: impl ToString,
        applicability: Applicability,
    ) -> &mut Self {
        self.span_suggestion_with_style(
            sp,
            msg,
            suggestion,
            applicability,
            SuggestionStyle::CompletelyHidden,
        );
        self
    } }

    /// Add a subdiagnostic from a type that implements `Subdiagnostic` (see
    /// [rustc_macros::Subdiagnostic]). Performs eager translation of any translatable messages
    /// used in the subdiagnostic, so suitable for use with repeated messages (i.e. re-use of
    /// interpolated variables).
    #[rustc_lint_diagnostics]
    pub fn subdiagnostic(&mut self, subdiagnostic: impl Subdiagnostic) -> &mut Self {
        subdiagnostic.add_to_diag(self);
        self
    }

    /// Fluent variables are not namespaced from each other, so when
    /// `Diagnostic`s and `Subdiagnostic`s use the same variable name,
    /// one value will clobber the other. Eagerly translating the
    /// diagnostic uses the variables defined right then, before the
    /// clobbering occurs.
    pub fn eagerly_translate(&self, msg: impl Into<SubdiagMessage>) -> SubdiagMessage {
        let args = self.args.iter();
        let msg = self.subdiagnostic_message_to_diagnostic_message(msg.into());
        self.dcx.eagerly_translate(msg, args)
    }

    with_fn! { with_span,
    /// Add a span.
    #[rustc_lint_diagnostics]
    pub fn span(&mut self, sp: impl Into<MultiSpan>) -> &mut Self {
        self.span = sp.into();
        if let Some(span) = self.span.primary_span() {
            self.sort_span = span;
        }
        self
    } }

    #[rustc_lint_diagnostics]
    pub fn is_lint(&mut self, name: String, has_future_breakage: bool) -> &mut Self {
        self.is_lint = Some(IsLint { name, has_future_breakage });
        self
    }

    with_fn! { with_code,
    /// Add an error code.
    #[rustc_lint_diagnostics]
    pub fn code(&mut self, code: ErrCode) -> &mut Self {
        self.code = Some(code);
        self
    } }

    with_fn! { with_lint_id,
    /// Add an argument.
    #[rustc_lint_diagnostics]
    pub fn lint_id(
        &mut self,
        id: LintExpectationId,
    ) -> &mut Self {
        self.lint_id = Some(id);
        self
    } }

    with_fn! { with_primary_message,
    /// Add a primary message.
    #[rustc_lint_diagnostics]
    pub fn primary_message(&mut self, msg: impl Into<DiagMessage>) -> &mut Self {
        self.messages[0] = (msg.into(), Style::NoStyle);
        self
    } }

    with_fn! { with_arg,
    /// Add an argument.
    #[rustc_lint_diagnostics]
    pub fn arg(
        &mut self,
        name: impl Into<DiagArgName>,
        arg: impl IntoDiagArg,
    ) -> &mut Self {
        self.deref_mut().arg(name, arg);
        self
    } }

    /// Helper function that takes a `SubdiagMessage` and returns a `DiagMessage` by
    /// combining it with the primary message of the diagnostic (if translatable, otherwise it just
    /// passes the user's string along).
    pub(crate) fn subdiagnostic_message_to_diagnostic_message(
        &self,
        attr: impl Into<SubdiagMessage>,
    ) -> DiagMessage {
        self.deref().subdiagnostic_message_to_diagnostic_message(attr)
    }

    /// Convenience function for internal use, clients should use one of the
    /// public methods above.
    ///
    /// Used by `proc_macro_server` for implementing `server::Diagnostic`.
    pub fn sub(&mut self, level: Level, message: impl Into<SubdiagMessage>, span: MultiSpan) {
        self.deref_mut().sub(level, message, span);
    }

    /// Convenience function for internal use, clients should use one of the
    /// public methods above.
    fn sub_with_highlights(&mut self, level: Level, messages: Vec<StringPart>, span: MultiSpan) {
        let messages = messages
            .into_iter()
            .map(|m| (self.subdiagnostic_message_to_diagnostic_message(m.content), m.style))
            .collect();
        let sub = Subdiag { level, messages, span };
        self.children.push(sub);
    }

    /// Takes the diagnostic. For use by methods that consume the Diag: `emit`,
    /// `cancel`, etc. Afterwards, `drop` is the only code that will be run on
    /// `self`.
    fn take_diag(&mut self) -> DiagInner {
        if let Some(path) = &self.long_ty_path {
            self.note(format!(
                "the full name for the type has been written to '{}'",
                path.display()
            ));
            self.note("consider using `--verbose` to print the full type name to the console");
        }
        *self.diag.take().unwrap()
    }

    /// This method allows us to access the path of the file where "long types" are written to.
    ///
    /// When calling `Diag::emit`, as part of that we will check if a `long_ty_path` has been set,
    /// and if it has been then we add a note mentioning the file where the "long types" were
    /// written to.
    ///
    /// When calling `tcx.short_string()` after a `Diag` is constructed, the preferred way of doing
    /// so is `tcx.short_string(ty, diag.long_ty_path())`. The diagnostic itself is the one that
    /// keeps the existence of a "long type" anywhere in the diagnostic, so the note telling the
    /// user where we wrote the file to is only printed once at most, *and* it makes it much harder
    /// to forget to set it.
    ///
    /// If the diagnostic hasn't been created before a "short ty string" is created, then you should
    /// ensure that this method is called to set it `*diag.long_ty_path() = path`.
    ///
    /// As a rule of thumb, if you see or add at least one `tcx.short_string()` call anywhere, in a
    /// scope, `diag.long_ty_path()` should be called once somewhere close by.
    pub fn long_ty_path(&mut self) -> &mut Option<PathBuf> {
        &mut self.long_ty_path
    }

    /// Most `emit_producing_guarantee` functions use this as a starting point.
    fn emit_producing_nothing(mut self) {
        let diag = self.take_diag();
        self.dcx.emit_diagnostic(diag);
    }

    /// `ErrorGuaranteed::emit_producing_guarantee` uses this.
    fn emit_producing_error_guaranteed(mut self) -> ErrorGuaranteed {
        let diag = self.take_diag();

        // The only error levels that produce `ErrorGuaranteed` are
        // `Error` and `DelayedBug`. But `DelayedBug` should never occur here
        // because delayed bugs have their level changed to `Bug` when they are
        // actually printed, so they produce an ICE.
        //
        // (Also, even though `level` isn't `pub`, the whole `DiagInner` could
        // be overwritten with a new one thanks to `DerefMut`. So this assert
        // protects against that, too.)
        assert!(
            matches!(diag.level, Level::Error | Level::DelayedBug),
            "invalid diagnostic level ({:?})",
            diag.level,
        );

        let guar = self.dcx.emit_diagnostic(diag);
        guar.unwrap()
    }

    /// Emit and consume the diagnostic.
    #[track_caller]
    pub fn emit(self) -> G::EmitResult {
        G::emit_producing_guarantee(self)
    }

    /// Emit the diagnostic unless `delay` is true,
    /// in which case the emission will be delayed as a bug.
    ///
    /// See `emit` and `delay_as_bug` for details.
    #[track_caller]
    pub fn emit_unless(mut self, delay: bool) -> G::EmitResult {
        if delay {
            self.downgrade_to_delayed_bug();
        }
        self.emit()
    }

    /// Cancel and consume the diagnostic. (A diagnostic must either be emitted or
    /// cancelled or it will panic when dropped).
    pub fn cancel(mut self) {
        self.diag = None;
        drop(self);
    }

    /// See `DiagCtxt::stash_diagnostic` for details.
    pub fn stash(mut self, span: Span, key: StashKey) -> Option<ErrorGuaranteed> {
        let diag = self.take_diag();
        self.dcx.stash_diagnostic(span, key, diag)
    }

    /// Delay emission of this diagnostic as a bug.
    ///
    /// This can be useful in contexts where an error indicates a bug but
    /// typically this only happens when other compilation errors have already
    /// happened. In those cases this can be used to defer emission of this
    /// diagnostic as a bug in the compiler only if no other errors have been
    /// emitted.
    ///
    /// In the meantime, though, callsites are required to deal with the "bug"
    /// locally in whichever way makes the most sense.
    #[track_caller]
    pub fn delay_as_bug(mut self) -> G::EmitResult {
        self.downgrade_to_delayed_bug();
        self.emit()
    }

    pub fn remove_arg(&mut self, name: &str) {
        if let Some(diag) = self.diag.as_mut() {
            diag.remove_arg(name);
        }
    }
}

/// Destructor bomb: every `Diag` must be consumed (emitted, cancelled, etc.)
/// or we emit a bug.
impl<G: EmissionGuarantee> Drop for Diag<'_, G> {
    fn drop(&mut self) {
        match self.diag.take() {
            Some(diag) if !panicking() => {
                self.dcx.emit_diagnostic(DiagInner::new(
                    Level::Bug,
                    DiagMessage::from("the following error was constructed but not emitted"),
                ));
                self.dcx.emit_diagnostic(*diag);
                panic!("error was constructed but not emitted");
            }
            _ => {}
        }
    }
}

#[macro_export]
macro_rules! struct_span_code_err {
    ($dcx:expr, $span:expr, $code:expr, $($message:tt)*) => ({
        $dcx.struct_span_err($span, format!($($message)*)).with_code($code)
    })
}
