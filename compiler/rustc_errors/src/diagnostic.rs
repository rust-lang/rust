use crate::snippet::Style;
use crate::{
    CodeSuggestion, DiagnosticBuilder, DiagnosticMessage, EmissionGuarantee, ErrCode, Level,
    MultiSpan, SubdiagnosticMessage, Substitution, SubstitutionPart, SuggestionStyle,
};
use rustc_data_structures::fx::FxIndexMap;
use rustc_error_messages::fluent_value_from_str_list_sep_by_and;
use rustc_error_messages::FluentValue;
use rustc_lint_defs::{Applicability, LintExpectationId};
use rustc_span::symbol::Symbol;
use rustc_span::{Span, DUMMY_SP};
use std::borrow::Cow;
use std::fmt::{self, Debug};
use std::hash::{Hash, Hasher};
use std::ops::{Deref, DerefMut};
use std::panic::Location;

/// Error type for `Diagnostic`'s `suggestions` field, indicating that
/// `.disable_suggestions()` was called on the `Diagnostic`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Encodable, Decodable)]
pub struct SuggestionsDisabled;

/// Simplified version of `FluentArg` that can implement `Encodable` and `Decodable`. Collection of
/// `DiagnosticArg` are converted to `FluentArgs` (consuming the collection) at the start of
/// diagnostic emission.
pub type DiagnosticArg<'iter> = (&'iter DiagnosticArgName, &'iter DiagnosticArgValue);

/// Name of a diagnostic argument.
pub type DiagnosticArgName = Cow<'static, str>;

/// Simplified version of `FluentValue` that can implement `Encodable` and `Decodable`. Converted
/// to a `FluentValue` by the emitter to be used in diagnostic translation.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Encodable, Decodable)]
pub enum DiagnosticArgValue {
    Str(Cow<'static, str>),
    // This gets converted to a `FluentNumber`, which is an `f64`. An `i32`
    // safely fits in an `f64`. Any integers bigger than that will be converted
    // to strings in `into_diagnostic_arg` and stored using the `Str` variant.
    Number(i32),
    StrListSepByAnd(Vec<Cow<'static, str>>),
}

/// Converts a value of a type into a `DiagnosticArg` (typically a field of an `IntoDiagnostic`
/// struct). Implemented as a custom trait rather than `From` so that it is implemented on the type
/// being converted rather than on `DiagnosticArgValue`, which enables types from other `rustc_*`
/// crates to implement this.
pub trait IntoDiagnosticArg {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue;
}

impl IntoDiagnosticArg for DiagnosticArgValue {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue {
        self
    }
}

impl Into<FluentValue<'static>> for DiagnosticArgValue {
    fn into(self) -> FluentValue<'static> {
        match self {
            DiagnosticArgValue::Str(s) => From::from(s),
            DiagnosticArgValue::Number(n) => From::from(n),
            DiagnosticArgValue::StrListSepByAnd(l) => fluent_value_from_str_list_sep_by_and(l),
        }
    }
}

/// Trait implemented by error types. This should not be implemented manually. Instead, use
/// `#[derive(Subdiagnostic)]` -- see [rustc_macros::Subdiagnostic].
#[rustc_diagnostic_item = "AddToDiagnostic"]
pub trait AddToDiagnostic
where
    Self: Sized,
{
    /// Add a subdiagnostic to an existing diagnostic.
    fn add_to_diagnostic<G: EmissionGuarantee>(self, diag: &mut DiagnosticBuilder<'_, G>) {
        self.add_to_diagnostic_with(diag, |_, m| m);
    }

    /// Add a subdiagnostic to an existing diagnostic where `f` is invoked on every message used
    /// (to optionally perform eager translation).
    fn add_to_diagnostic_with<G: EmissionGuarantee, F: SubdiagnosticMessageOp<G>>(
        self,
        diag: &mut DiagnosticBuilder<'_, G>,
        f: F,
    );
}

pub trait SubdiagnosticMessageOp<G> =
    Fn(&mut DiagnosticBuilder<'_, G>, SubdiagnosticMessage) -> SubdiagnosticMessage;

/// Trait implemented by lint types. This should not be implemented manually. Instead, use
/// `#[derive(LintDiagnostic)]` -- see [rustc_macros::LintDiagnostic].
#[rustc_diagnostic_item = "DecorateLint"]
pub trait DecorateLint<'a, G: EmissionGuarantee> {
    /// Decorate and emit a lint.
    fn decorate_lint<'b>(self, diag: &'b mut DiagnosticBuilder<'a, G>);

    fn msg(&self) -> DiagnosticMessage;
}

/// The main part of a diagnostic. Note that `DiagnosticBuilder`, which wraps
/// this type, is used for most operations, and should be used instead whenever
/// possible. This type should only be used when `DiagnosticBuilder`'s lifetime
/// causes difficulties, e.g. when storing diagnostics within `DiagCtxt`.
#[must_use]
#[derive(Clone, Debug, Encodable, Decodable)]
pub struct Diagnostic {
    // NOTE(eddyb) this is private to disallow arbitrary after-the-fact changes,
    // outside of what methods in this crate themselves allow.
    pub(crate) level: Level,

    pub messages: Vec<(DiagnosticMessage, Style)>,
    pub code: Option<ErrCode>,
    pub span: MultiSpan,
    pub children: Vec<SubDiagnostic>,
    pub suggestions: Result<Vec<CodeSuggestion>, SuggestionsDisabled>,
    args: FxIndexMap<DiagnosticArgName, DiagnosticArgValue>,

    /// This is not used for highlighting or rendering any error message. Rather, it can be used
    /// as a sort key to sort a buffer of diagnostics. By default, it is the primary span of
    /// `span` if there is one. Otherwise, it is `DUMMY_SP`.
    pub sort_span: Span,

    pub is_lint: Option<IsLint>,

    /// With `-Ztrack_diagnostics` enabled,
    /// we print where in rustc this error was emitted.
    pub(crate) emitted_at: DiagnosticLocation,
}

#[derive(Clone, Debug, Encodable, Decodable)]
pub struct DiagnosticLocation {
    file: Cow<'static, str>,
    line: u32,
    col: u32,
}

impl DiagnosticLocation {
    #[track_caller]
    fn caller() -> Self {
        let loc = Location::caller();
        DiagnosticLocation { file: loc.file().into(), line: loc.line(), col: loc.column() }
    }
}

impl fmt::Display for DiagnosticLocation {
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

/// A "sub"-diagnostic attached to a parent diagnostic.
/// For example, a note attached to an error.
#[derive(Clone, Debug, PartialEq, Hash, Encodable, Decodable)]
pub struct SubDiagnostic {
    pub level: Level,
    pub messages: Vec<(DiagnosticMessage, Style)>,
    pub span: MultiSpan,
}

#[derive(Debug, PartialEq, Eq)]
pub struct DiagnosticStyledString(pub Vec<StringPart>);

impl DiagnosticStyledString {
    pub fn new() -> DiagnosticStyledString {
        DiagnosticStyledString(vec![])
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
    pub fn normal<S: Into<String>>(t: S) -> DiagnosticStyledString {
        DiagnosticStyledString(vec![StringPart::normal(t)])
    }

    pub fn highlighted<S: Into<String>>(t: S) -> DiagnosticStyledString {
        DiagnosticStyledString(vec![StringPart::highlighted(t)])
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

impl Diagnostic {
    #[track_caller]
    pub fn new<M: Into<DiagnosticMessage>>(level: Level, message: M) -> Self {
        Diagnostic::new_with_messages(level, vec![(message.into(), Style::NoStyle)])
    }

    #[track_caller]
    pub fn new_with_messages(level: Level, messages: Vec<(DiagnosticMessage, Style)>) -> Self {
        Diagnostic {
            level,
            messages,
            code: None,
            span: MultiSpan::new(),
            children: vec![],
            suggestions: Ok(vec![]),
            args: Default::default(),
            sort_span: DUMMY_SP,
            is_lint: None,
            emitted_at: DiagnosticLocation::caller(),
        }
    }

    #[inline(always)]
    pub fn level(&self) -> Level {
        self.level
    }

    pub fn is_error(&self) -> bool {
        match self.level {
            Level::Bug | Level::Fatal | Level::Error | Level::DelayedBug => true,

            Level::ForceWarning(_)
            | Level::Warning
            | Level::Note
            | Level::OnceNote
            | Level::Help
            | Level::OnceHelp
            | Level::FailureNote
            | Level::Allow
            | Level::Expect(_) => false,
        }
    }

    pub(crate) fn update_unstable_expectation_id(
        &mut self,
        unstable_to_stable: &FxIndexMap<LintExpectationId, LintExpectationId>,
    ) {
        if let Level::Expect(expectation_id) | Level::ForceWarning(Some(expectation_id)) =
            &mut self.level
        {
            if expectation_id.is_stable() {
                return;
            }

            // The unstable to stable map only maps the unstable `AttrId` to a stable `HirId` with an attribute index.
            // The lint index inside the attribute is manually transferred here.
            let lint_index = expectation_id.get_lint_index();
            expectation_id.set_lint_index(None);
            let mut stable_id = unstable_to_stable
                .get(expectation_id)
                .expect("each unstable `LintExpectationId` must have a matching stable id")
                .normalize();

            stable_id.set_lint_index(lint_index);
            *expectation_id = stable_id;
        }
    }

    /// Indicates whether this diagnostic should show up in cargo's future breakage report.
    pub(crate) fn has_future_breakage(&self) -> bool {
        matches!(self.is_lint, Some(IsLint { has_future_breakage: true, .. }))
    }

    pub(crate) fn is_force_warn(&self) -> bool {
        match self.level {
            Level::ForceWarning(_) => {
                assert!(self.is_lint.is_some());
                true
            }
            _ => false,
        }
    }

    // See comment on `DiagnosticBuilder::subdiagnostic_message_to_diagnostic_message`.
    pub(crate) fn subdiagnostic_message_to_diagnostic_message(
        &self,
        attr: impl Into<SubdiagnosticMessage>,
    ) -> DiagnosticMessage {
        let msg =
            self.messages.iter().map(|(msg, _)| msg).next().expect("diagnostic with no messages");
        msg.with_subdiagnostic_message(attr.into())
    }

    pub(crate) fn sub(
        &mut self,
        level: Level,
        message: impl Into<SubdiagnosticMessage>,
        span: MultiSpan,
    ) {
        let sub = SubDiagnostic {
            level,
            messages: vec![(
                self.subdiagnostic_message_to_diagnostic_message(message),
                Style::NoStyle,
            )],
            span,
        };
        self.children.push(sub);
    }

    pub(crate) fn arg(&mut self, name: impl Into<DiagnosticArgName>, arg: impl IntoDiagnosticArg) {
        self.args.insert(name.into(), arg.into_diagnostic_arg());
    }

    pub fn args(&self) -> impl Iterator<Item = DiagnosticArg<'_>> {
        self.args.iter()
    }

    pub fn replace_args(&mut self, args: FxIndexMap<DiagnosticArgName, DiagnosticArgValue>) {
        self.args = args;
    }
}

/// `DiagnosticBuilder` impls many `&mut self -> &mut Self` methods. Each one
/// modifies an existing diagnostic, either in a standalone fashion, e.g.
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
        #[doc = concat!("See [`DiagnosticBuilder::", stringify!($f), "()`].")]
        pub fn $f(&mut $self, $($name: $ty),*) -> &mut Self {
            $($body)*
        }

        // The `with_*` variant.
        $(#[$attrs])*
        #[doc = concat!("See [`DiagnosticBuilder::", stringify!($f), "()`].")]
        pub fn $with_f(mut $self, $($name: $ty),*) -> Self {
            $self.$f($($name),*);
            $self
        }
    };
}

impl<'a, G: EmissionGuarantee> DiagnosticBuilder<'a, G> {
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
    pub fn span_label(&mut self, span: Span, label: impl Into<SubdiagnosticMessage>) -> &mut Self {
        let msg = self.subdiagnostic_message_to_diagnostic_message(label);
        self.span.push_span_label(span, msg);
        self
    } }

    with_fn! { with_span_labels,
    /// Labels all the given spans with the provided label.
    /// See [`Self::span_label()`] for more information.
    pub fn span_labels(&mut self, spans: impl IntoIterator<Item = Span>, label: &str) -> &mut Self {
        for span in spans {
            self.span_label(span, label.to_string());
        }
        self
    } }

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

    pub fn note_expected_found(
        &mut self,
        expected_label: &dyn fmt::Display,
        expected: DiagnosticStyledString,
        found_label: &dyn fmt::Display,
        found: DiagnosticStyledString,
    ) -> &mut Self {
        self.note_expected_found_extra(expected_label, expected, found_label, found, &"", &"")
    }

    pub fn note_expected_found_extra(
        &mut self,
        expected_label: &dyn fmt::Display,
        expected: DiagnosticStyledString,
        found_label: &dyn fmt::Display,
        found: DiagnosticStyledString,
        expected_extra: &dyn fmt::Display,
        found_extra: &dyn fmt::Display,
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
        msg.extend(expected.0.into_iter());
        msg.push(StringPart::normal(format!("`{expected_extra}\n")));
        msg.push(StringPart::normal(format!("{}{} `", " ".repeat(found_padding), found_label)));
        msg.extend(found.0.into_iter());
        msg.push(StringPart::normal(format!("`{found_extra}")));

        // For now, just attach these as notes.
        self.highlighted_note(msg);
        self
    }

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
    pub fn note(&mut self, msg: impl Into<SubdiagnosticMessage>) -> &mut Self {
        self.sub(Level::Note, msg, MultiSpan::new());
        self
    } }

    fn highlighted_note(&mut self, msg: Vec<StringPart>) -> &mut Self {
        self.sub_with_highlights(Level::Note, msg, MultiSpan::new());
        self
    }

    /// This is like [`DiagnosticBuilder::note()`], but it's only printed once.
    pub fn note_once(&mut self, msg: impl Into<SubdiagnosticMessage>) -> &mut Self {
        self.sub(Level::OnceNote, msg, MultiSpan::new());
        self
    }

    with_fn! { with_span_note,
    /// Prints the span with a note above it.
    /// This is like [`DiagnosticBuilder::note()`], but it gets its own span.
    #[rustc_lint_diagnostics]
    pub fn span_note(
        &mut self,
        sp: impl Into<MultiSpan>,
        msg: impl Into<SubdiagnosticMessage>,
    ) -> &mut Self {
        self.sub(Level::Note, msg, sp.into());
        self
    } }

    /// Prints the span with a note above it.
    /// This is like [`DiagnosticBuilder::note_once()`], but it gets its own span.
    pub fn span_note_once<S: Into<MultiSpan>>(
        &mut self,
        sp: S,
        msg: impl Into<SubdiagnosticMessage>,
    ) -> &mut Self {
        self.sub(Level::OnceNote, msg, sp.into());
        self
    }

    with_fn! { with_warn,
    /// Add a warning attached to this diagnostic.
    #[rustc_lint_diagnostics]
    pub fn warn(&mut self, msg: impl Into<SubdiagnosticMessage>) -> &mut Self {
        self.sub(Level::Warning, msg, MultiSpan::new());
        self
    } }

    /// Prints the span with a warning above it.
    /// This is like [`DiagnosticBuilder::warn()`], but it gets its own span.
    #[rustc_lint_diagnostics]
    pub fn span_warn<S: Into<MultiSpan>>(
        &mut self,
        sp: S,
        msg: impl Into<SubdiagnosticMessage>,
    ) -> &mut Self {
        self.sub(Level::Warning, msg, sp.into());
        self
    }

    with_fn! { with_help,
    /// Add a help message attached to this diagnostic.
    #[rustc_lint_diagnostics]
    pub fn help(&mut self, msg: impl Into<SubdiagnosticMessage>) -> &mut Self {
        self.sub(Level::Help, msg, MultiSpan::new());
        self
    } }

    /// This is like [`DiagnosticBuilder::help()`], but it's only printed once.
    pub fn help_once(&mut self, msg: impl Into<SubdiagnosticMessage>) -> &mut Self {
        self.sub(Level::OnceHelp, msg, MultiSpan::new());
        self
    }

    /// Add a help message attached to this diagnostic with a customizable highlighted message.
    pub fn highlighted_help(&mut self, msg: Vec<StringPart>) -> &mut Self {
        self.sub_with_highlights(Level::Help, msg, MultiSpan::new());
        self
    }

    /// Prints the span with some help above it.
    /// This is like [`DiagnosticBuilder::help()`], but it gets its own span.
    #[rustc_lint_diagnostics]
    pub fn span_help<S: Into<MultiSpan>>(
        &mut self,
        sp: S,
        msg: impl Into<SubdiagnosticMessage>,
    ) -> &mut Self {
        self.sub(Level::Help, msg, sp.into());
        self
    }

    /// Disallow attaching suggestions this diagnostic.
    /// Any suggestions attached e.g. with the `span_suggestion_*` methods
    /// (before and after the call to `disable_suggestions`) will be ignored.
    pub fn disable_suggestions(&mut self) -> &mut Self {
        self.suggestions = Err(SuggestionsDisabled);
        self
    }

    /// Helper for pushing to `self.suggestions`, if available (not disable).
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

        if let Ok(suggestions) = &mut self.suggestions {
            suggestions.push(suggestion);
        }
    }

    with_fn! { with_multipart_suggestion,
    /// Show a suggestion that has multiple parts to it.
    /// In other words, multiple changes need to be applied as part of this suggestion.
    pub fn multipart_suggestion(
        &mut self,
        msg: impl Into<SubdiagnosticMessage>,
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

    /// Show a suggestion that has multiple parts to it, always as it's own subdiagnostic.
    /// In other words, multiple changes need to be applied as part of this suggestion.
    pub fn multipart_suggestion_verbose(
        &mut self,
        msg: impl Into<SubdiagnosticMessage>,
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

    /// [`DiagnosticBuilder::multipart_suggestion()`] but you can set the [`SuggestionStyle`].
    pub fn multipart_suggestion_with_style(
        &mut self,
        msg: impl Into<SubdiagnosticMessage>,
        mut suggestion: Vec<(Span, String)>,
        applicability: Applicability,
        style: SuggestionStyle,
    ) -> &mut Self {
        suggestion.sort_unstable();
        suggestion.dedup();

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
    pub fn tool_only_multipart_suggestion(
        &mut self,
        msg: impl Into<SubdiagnosticMessage>,
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
    /// See `CodeSuggestion` for more information.
    pub fn span_suggestion(
        &mut self,
        sp: Span,
        msg: impl Into<SubdiagnosticMessage>,
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

    /// [`DiagnosticBuilder::span_suggestion()`] but you can set the [`SuggestionStyle`].
    pub fn span_suggestion_with_style(
        &mut self,
        sp: Span,
        msg: impl Into<SubdiagnosticMessage>,
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
    pub fn span_suggestion_verbose(
        &mut self,
        sp: Span,
        msg: impl Into<SubdiagnosticMessage>,
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
    /// See also [`DiagnosticBuilder::span_suggestion()`].
    pub fn span_suggestions(
        &mut self,
        sp: Span,
        msg: impl Into<SubdiagnosticMessage>,
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

    pub fn span_suggestions_with_style(
        &mut self,
        sp: Span,
        msg: impl Into<SubdiagnosticMessage>,
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
    /// See also [`DiagnosticBuilder::multipart_suggestion()`].
    pub fn multipart_suggestions(
        &mut self,
        msg: impl Into<SubdiagnosticMessage>,
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
    /// See `CodeSuggestion` for more information.
    pub fn span_suggestion_short(
        &mut self,
        sp: Span,
        msg: impl Into<SubdiagnosticMessage>,
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
    pub fn span_suggestion_hidden(
        &mut self,
        sp: Span,
        msg: impl Into<SubdiagnosticMessage>,
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
        msg: impl Into<SubdiagnosticMessage>,
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
    pub fn subdiagnostic(
        &mut self,
        dcx: &crate::DiagCtxt,
        subdiagnostic: impl AddToDiagnostic,
    ) -> &mut Self {
        subdiagnostic.add_to_diagnostic_with(self, |diag, msg| {
            let args = diag.args();
            let msg = diag.subdiagnostic_message_to_diagnostic_message(msg);
            dcx.eagerly_translate(msg, args)
        });
        self
    }

    with_fn! { with_span,
    /// Add a span.
    pub fn span(&mut self, sp: impl Into<MultiSpan>) -> &mut Self {
        self.span = sp.into();
        if let Some(span) = self.span.primary_span() {
            self.sort_span = span;
        }
        self
    } }

    pub fn is_lint(&mut self, name: String, has_future_breakage: bool) -> &mut Self {
        self.is_lint = Some(IsLint { name, has_future_breakage });
        self
    }

    with_fn! { with_code,
    /// Add an error code.
    pub fn code(&mut self, code: ErrCode) -> &mut Self {
        self.code = Some(code);
        self
    } }

    with_fn! { with_primary_message,
    /// Add a primary message.
    pub fn primary_message(&mut self, msg: impl Into<DiagnosticMessage>) -> &mut Self {
        self.messages[0] = (msg.into(), Style::NoStyle);
        self
    } }

    with_fn! { with_arg,
    /// Add an argument.
    pub fn arg(
        &mut self,
        name: impl Into<DiagnosticArgName>,
        arg: impl IntoDiagnosticArg,
    ) -> &mut Self {
        self.deref_mut().arg(name, arg);
        self
    } }

    /// Helper function that takes a `SubdiagnosticMessage` and returns a `DiagnosticMessage` by
    /// combining it with the primary message of the diagnostic (if translatable, otherwise it just
    /// passes the user's string along).
    pub(crate) fn subdiagnostic_message_to_diagnostic_message(
        &self,
        attr: impl Into<SubdiagnosticMessage>,
    ) -> DiagnosticMessage {
        self.deref().subdiagnostic_message_to_diagnostic_message(attr)
    }

    /// Convenience function for internal use, clients should use one of the
    /// public methods above.
    ///
    /// Used by `proc_macro_server` for implementing `server::Diagnostic`.
    pub fn sub(&mut self, level: Level, message: impl Into<SubdiagnosticMessage>, span: MultiSpan) {
        self.deref_mut().sub(level, message, span);
    }

    /// Convenience function for internal use, clients should use one of the
    /// public methods above.
    fn sub_with_highlights(&mut self, level: Level, messages: Vec<StringPart>, span: MultiSpan) {
        let messages = messages
            .into_iter()
            .map(|m| (self.subdiagnostic_message_to_diagnostic_message(m.content), m.style))
            .collect();
        let sub = SubDiagnostic { level, messages, span };
        self.children.push(sub);
    }
}

impl Diagnostic {
    /// Fields used for Hash, and PartialEq trait
    fn keys(
        &self,
    ) -> (
        &Level,
        &[(DiagnosticMessage, Style)],
        &Option<ErrCode>,
        &MultiSpan,
        &[SubDiagnostic],
        &Result<Vec<CodeSuggestion>, SuggestionsDisabled>,
        Vec<(&DiagnosticArgName, &DiagnosticArgValue)>,
        &Option<IsLint>,
    ) {
        (
            &self.level,
            &self.messages,
            &self.code,
            &self.span,
            &self.children,
            &self.suggestions,
            self.args().collect(),
            // omit self.sort_span
            &self.is_lint,
            // omit self.emitted_at
        )
    }
}

impl Hash for Diagnostic {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        self.keys().hash(state);
    }
}

impl PartialEq for Diagnostic {
    fn eq(&self, other: &Self) -> bool {
        self.keys() == other.keys()
    }
}
