use crate::snippet::Style;
use crate::{
    CodeSuggestion, DiagnosticMessage, EmissionGuarantee, Level, LintDiagnosticBuilder, MultiSpan,
    SubdiagnosticMessage, Substitution, SubstitutionPart, SuggestionStyle,
};
use rustc_data_structures::fx::FxHashMap;
use rustc_error_messages::FluentValue;
use rustc_hir as hir;
use rustc_lint_defs::{Applicability, LintExpectationId};
use rustc_span::edition::LATEST_STABLE_EDITION;
use rustc_span::symbol::{Ident, MacroRulesNormalizedIdent, Symbol};
use rustc_span::{edition::Edition, Span, DUMMY_SP};
use std::borrow::Cow;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

/// Error type for `Diagnostic`'s `suggestions` field, indicating that
/// `.disable_suggestions()` was called on the `Diagnostic`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Encodable, Decodable)]
pub struct SuggestionsDisabled;

/// Simplified version of `FluentArg` that can implement `Encodable` and `Decodable`. Collection of
/// `DiagnosticArg` are converted to `FluentArgs` (consuming the collection) at the start of
/// diagnostic emission.
pub type DiagnosticArg<'source> = (Cow<'source, str>, DiagnosticArgValue<'source>);

/// Simplified version of `FluentValue` that can implement `Encodable` and `Decodable`. Converted
/// to a `FluentValue` by the emitter to be used in diagnostic translation.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Encodable, Decodable)]
pub enum DiagnosticArgValue<'source> {
    Str(Cow<'source, str>),
    Number(usize),
}

/// Converts a value of a type into a `DiagnosticArg` (typically a field of a `SessionDiagnostic`
/// struct). Implemented as a custom trait rather than `From` so that it is implemented on the type
/// being converted rather than on `DiagnosticArgValue`, which enables types from other `rustc_*`
/// crates to implement this.
pub trait IntoDiagnosticArg {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static>;
}

pub struct DiagnosticArgFromDisplay<'a>(pub &'a dyn fmt::Display);

impl IntoDiagnosticArg for DiagnosticArgFromDisplay<'_> {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        self.0.to_string().into_diagnostic_arg()
    }
}

impl<'a> From<&'a dyn fmt::Display> for DiagnosticArgFromDisplay<'a> {
    fn from(t: &'a dyn fmt::Display) -> Self {
        DiagnosticArgFromDisplay(t)
    }
}

impl<'a, T: fmt::Display> From<&'a T> for DiagnosticArgFromDisplay<'a> {
    fn from(t: &'a T) -> Self {
        DiagnosticArgFromDisplay(t)
    }
}

macro_rules! into_diagnostic_arg_using_display {
    ($( $ty:ty ),+ $(,)?) => {
        $(
            impl IntoDiagnosticArg for $ty {
                fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
                    self.to_string().into_diagnostic_arg()
                }
            }
        )+
    }
}

into_diagnostic_arg_using_display!(
    i8,
    u8,
    i16,
    u16,
    i32,
    u32,
    i64,
    u64,
    i128,
    u128,
    std::io::Error,
    std::num::NonZeroU32,
    hir::Target,
    Edition,
    Ident,
    MacroRulesNormalizedIdent,
);

impl IntoDiagnosticArg for bool {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        if self {
            DiagnosticArgValue::Str(Cow::Borrowed("true"))
        } else {
            DiagnosticArgValue::Str(Cow::Borrowed("false"))
        }
    }
}

impl IntoDiagnosticArg for char {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        DiagnosticArgValue::Str(Cow::Owned(format!("{:?}", self)))
    }
}

impl IntoDiagnosticArg for Symbol {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        self.to_ident_string().into_diagnostic_arg()
    }
}

impl<'a> IntoDiagnosticArg for &'a str {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        self.to_string().into_diagnostic_arg()
    }
}

impl IntoDiagnosticArg for String {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        DiagnosticArgValue::Str(Cow::Owned(self))
    }
}

impl<'a> IntoDiagnosticArg for &'a Path {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        DiagnosticArgValue::Str(Cow::Owned(self.display().to_string()))
    }
}

impl IntoDiagnosticArg for PathBuf {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        DiagnosticArgValue::Str(Cow::Owned(self.display().to_string()))
    }
}

impl IntoDiagnosticArg for usize {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        DiagnosticArgValue::Number(self)
    }
}

impl<'source> Into<FluentValue<'source>> for DiagnosticArgValue<'source> {
    fn into(self) -> FluentValue<'source> {
        match self {
            DiagnosticArgValue::Str(s) => From::from(s),
            DiagnosticArgValue::Number(n) => From::from(n),
        }
    }
}

impl IntoDiagnosticArg for hir::ConstContext {
    fn into_diagnostic_arg(self) -> DiagnosticArgValue<'static> {
        DiagnosticArgValue::Str(Cow::Borrowed(match self {
            hir::ConstContext::ConstFn => "constant function",
            hir::ConstContext::Static(_) => "static",
            hir::ConstContext::Const => "constant",
        }))
    }
}

/// Trait implemented by error types. This should not be implemented manually. Instead, use
/// `#[derive(SessionSubdiagnostic)]` -- see [rustc_macros::SessionSubdiagnostic].
#[rustc_diagnostic_item = "AddSubdiagnostic"]
pub trait AddSubdiagnostic {
    /// Add a subdiagnostic to an existing diagnostic.
    fn add_to_diagnostic(self, diag: &mut Diagnostic);
}

/// Trait implemented by lint types. This should not be implemented manually. Instead, use
/// `#[derive(LintDiagnostic)]` -- see [rustc_macros::LintDiagnostic].
#[rustc_diagnostic_item = "DecorateLint"]
pub trait DecorateLint<'a, G: EmissionGuarantee> {
    /// Decorate and emit a lint.
    fn decorate_lint(self, diag: LintDiagnosticBuilder<'a, G>);
}

#[must_use]
#[derive(Clone, Debug, Encodable, Decodable)]
pub struct Diagnostic {
    // NOTE(eddyb) this is private to disallow arbitrary after-the-fact changes,
    // outside of what methods in this crate themselves allow.
    pub(crate) level: Level,

    pub message: Vec<(DiagnosticMessage, Style)>,
    pub code: Option<DiagnosticId>,
    pub span: MultiSpan,
    pub children: Vec<SubDiagnostic>,
    pub suggestions: Result<Vec<CodeSuggestion>, SuggestionsDisabled>,
    args: Vec<DiagnosticArg<'static>>,

    /// This is not used for highlighting or rendering any error message.  Rather, it can be used
    /// as a sort key to sort a buffer of diagnostics.  By default, it is the primary span of
    /// `span` if there is one.  Otherwise, it is `DUMMY_SP`.
    pub sort_span: Span,

    /// If diagnostic is from Lint, custom hash function ignores notes
    /// otherwise hash is based on the all the fields
    pub is_lint: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Encodable, Decodable)]
pub enum DiagnosticId {
    Error(String),
    Lint { name: String, has_future_breakage: bool, is_force_warn: bool },
}

/// A "sub"-diagnostic attached to a parent diagnostic.
/// For example, a note attached to an error.
#[derive(Clone, Debug, PartialEq, Hash, Encodable, Decodable)]
pub struct SubDiagnostic {
    pub level: Level,
    pub message: Vec<(DiagnosticMessage, Style)>,
    pub span: MultiSpan,
    pub render_span: Option<MultiSpan>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct DiagnosticStyledString(pub Vec<StringPart>);

impl DiagnosticStyledString {
    pub fn new() -> DiagnosticStyledString {
        DiagnosticStyledString(vec![])
    }
    pub fn push_normal<S: Into<String>>(&mut self, t: S) {
        self.0.push(StringPart::Normal(t.into()));
    }
    pub fn push_highlighted<S: Into<String>>(&mut self, t: S) {
        self.0.push(StringPart::Highlighted(t.into()));
    }
    pub fn push<S: Into<String>>(&mut self, t: S, highlight: bool) {
        if highlight {
            self.push_highlighted(t);
        } else {
            self.push_normal(t);
        }
    }
    pub fn normal<S: Into<String>>(t: S) -> DiagnosticStyledString {
        DiagnosticStyledString(vec![StringPart::Normal(t.into())])
    }

    pub fn highlighted<S: Into<String>>(t: S) -> DiagnosticStyledString {
        DiagnosticStyledString(vec![StringPart::Highlighted(t.into())])
    }

    pub fn content(&self) -> String {
        self.0.iter().map(|x| x.content()).collect::<String>()
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum StringPart {
    Normal(String),
    Highlighted(String),
}

impl StringPart {
    pub fn content(&self) -> &str {
        match self {
            &StringPart::Normal(ref s) | &StringPart::Highlighted(ref s) => s,
        }
    }
}

impl Diagnostic {
    pub fn new<M: Into<DiagnosticMessage>>(level: Level, message: M) -> Self {
        Diagnostic::new_with_code(level, None, message)
    }

    pub fn new_with_code<M: Into<DiagnosticMessage>>(
        level: Level,
        code: Option<DiagnosticId>,
        message: M,
    ) -> Self {
        Diagnostic {
            level,
            message: vec![(message.into(), Style::NoStyle)],
            code,
            span: MultiSpan::new(),
            children: vec![],
            suggestions: Ok(vec![]),
            args: vec![],
            sort_span: DUMMY_SP,
            is_lint: false,
        }
    }

    #[inline(always)]
    pub fn level(&self) -> Level {
        self.level
    }

    pub fn is_error(&self) -> bool {
        match self.level {
            Level::Bug
            | Level::DelayedBug
            | Level::Fatal
            | Level::Error { .. }
            | Level::FailureNote => true,

            Level::Warning(_)
            | Level::Note
            | Level::OnceNote
            | Level::Help
            | Level::Allow
            | Level::Expect(_) => false,
        }
    }

    pub fn update_unstable_expectation_id(
        &mut self,
        unstable_to_stable: &FxHashMap<LintExpectationId, LintExpectationId>,
    ) {
        if let Level::Expect(expectation_id) | Level::Warning(Some(expectation_id)) =
            &mut self.level
        {
            if expectation_id.is_stable() {
                return;
            }

            // The unstable to stable map only maps the unstable `AttrId` to a stable `HirId` with an attribute index.
            // The lint index inside the attribute is manually transferred here.
            let lint_index = expectation_id.get_lint_index();
            expectation_id.set_lint_index(None);
            let mut stable_id = *unstable_to_stable
                .get(&expectation_id)
                .expect("each unstable `LintExpectationId` must have a matching stable id");

            stable_id.set_lint_index(lint_index);
            *expectation_id = stable_id;
        }
    }

    pub fn has_future_breakage(&self) -> bool {
        match self.code {
            Some(DiagnosticId::Lint { has_future_breakage, .. }) => has_future_breakage,
            _ => false,
        }
    }

    pub fn is_force_warn(&self) -> bool {
        match self.code {
            Some(DiagnosticId::Lint { is_force_warn, .. }) => is_force_warn,
            _ => false,
        }
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
    pub fn downgrade_to_delayed_bug(&mut self) -> &mut Self {
        assert!(
            self.is_error(),
            "downgrade_to_delayed_bug: cannot downgrade {:?} to DelayedBug: not an error",
            self.level
        );
        self.level = Level::DelayedBug;

        self
    }

    /// Adds a span/label to be included in the resulting snippet.
    ///
    /// This is pushed onto the [`MultiSpan`] that was created when the diagnostic
    /// was first built. That means it will be shown together with the original
    /// span/label, *not* a span added by one of the `span_{note,warn,help,suggestions}` methods.
    ///
    /// This span is *not* considered a ["primary span"][`MultiSpan`]; only
    /// the `Span` supplied when creating the diagnostic is primary.
    #[rustc_lint_diagnostics]
    pub fn span_label(&mut self, span: Span, label: impl Into<SubdiagnosticMessage>) -> &mut Self {
        self.span.push_span_label(span, self.subdiagnostic_message_to_diagnostic_message(label));
        self
    }

    /// Labels all the given spans with the provided label.
    /// See [`Self::span_label()`] for more information.
    pub fn span_labels(
        &mut self,
        spans: impl IntoIterator<Item = Span>,
        label: impl AsRef<str>,
    ) -> &mut Self {
        let label = label.as_ref();
        for span in spans {
            self.span_label(span, label);
        }
        self
    }

    pub fn replace_span_with(&mut self, after: Span) -> &mut Self {
        let before = self.span.clone();
        self.set_span(after);
        for span_label in before.span_labels() {
            if let Some(label) = span_label.label {
                self.span.push_span_label(after, label);
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

    pub fn note_unsuccessful_coercion(
        &mut self,
        expected: DiagnosticStyledString,
        found: DiagnosticStyledString,
    ) -> &mut Self {
        let mut msg: Vec<_> = vec![("required when trying to coerce from type `", Style::NoStyle)];
        msg.extend(expected.0.iter().map(|x| match *x {
            StringPart::Normal(ref s) => (s.as_str(), Style::NoStyle),
            StringPart::Highlighted(ref s) => (s.as_str(), Style::Highlight),
        }));
        msg.push(("` to type '", Style::NoStyle));
        msg.extend(found.0.iter().map(|x| match *x {
            StringPart::Normal(ref s) => (s.as_str(), Style::NoStyle),
            StringPart::Highlighted(ref s) => (s.as_str(), Style::Highlight),
        }));
        msg.push(("`", Style::NoStyle));

        // For now, just attach these as notes
        self.highlighted_note(msg);
        self
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
            format!("expected {}", expected_label)
        };
        let found_label = found_label.to_string();
        let found_label = if found_label.is_empty() {
            "found".to_string()
        } else {
            format!("found {}", found_label)
        };
        let (found_padding, expected_padding) = if expected_label.len() > found_label.len() {
            (expected_label.len() - found_label.len(), 0)
        } else {
            (0, found_label.len() - expected_label.len())
        };
        let mut msg: Vec<_> =
            vec![(format!("{}{} `", " ".repeat(expected_padding), expected_label), Style::NoStyle)];
        msg.extend(expected.0.iter().map(|x| match *x {
            StringPart::Normal(ref s) => (s.to_owned(), Style::NoStyle),
            StringPart::Highlighted(ref s) => (s.to_owned(), Style::Highlight),
        }));
        msg.push((format!("`{}\n", expected_extra), Style::NoStyle));
        msg.push((format!("{}{} `", " ".repeat(found_padding), found_label), Style::NoStyle));
        msg.extend(found.0.iter().map(|x| match *x {
            StringPart::Normal(ref s) => (s.to_owned(), Style::NoStyle),
            StringPart::Highlighted(ref s) => (s.to_owned(), Style::Highlight),
        }));
        msg.push((format!("`{}", found_extra), Style::NoStyle));

        // For now, just attach these as notes.
        self.highlighted_note(msg);
        self
    }

    pub fn note_trait_signature(&mut self, name: Symbol, signature: String) -> &mut Self {
        self.highlighted_note(vec![
            (format!("`{}` from trait: `", name), Style::NoStyle),
            (signature, Style::Highlight),
            ("`".to_string(), Style::NoStyle),
        ]);
        self
    }

    /// Add a note attached to this diagnostic.
    #[rustc_lint_diagnostics]
    pub fn note(&mut self, msg: impl Into<SubdiagnosticMessage>) -> &mut Self {
        self.sub(Level::Note, msg, MultiSpan::new(), None);
        self
    }

    pub fn highlighted_note<M: Into<SubdiagnosticMessage>>(
        &mut self,
        msg: Vec<(M, Style)>,
    ) -> &mut Self {
        self.sub_with_highlights(Level::Note, msg, MultiSpan::new(), None);
        self
    }

    /// Prints the span with a note above it.
    /// This is like [`Diagnostic::note()`], but it gets its own span.
    pub fn note_once(&mut self, msg: impl Into<SubdiagnosticMessage>) -> &mut Self {
        self.sub(Level::OnceNote, msg, MultiSpan::new(), None);
        self
    }

    /// Prints the span with a note above it.
    /// This is like [`Diagnostic::note()`], but it gets its own span.
    #[rustc_lint_diagnostics]
    pub fn span_note<S: Into<MultiSpan>>(
        &mut self,
        sp: S,
        msg: impl Into<SubdiagnosticMessage>,
    ) -> &mut Self {
        self.sub(Level::Note, msg, sp.into(), None);
        self
    }

    /// Prints the span with a note above it.
    /// This is like [`Diagnostic::note()`], but it gets its own span.
    pub fn span_note_once<S: Into<MultiSpan>>(
        &mut self,
        sp: S,
        msg: impl Into<SubdiagnosticMessage>,
    ) -> &mut Self {
        self.sub(Level::OnceNote, msg, sp.into(), None);
        self
    }

    /// Add a warning attached to this diagnostic.
    #[rustc_lint_diagnostics]
    pub fn warn(&mut self, msg: impl Into<SubdiagnosticMessage>) -> &mut Self {
        self.sub(Level::Warning(None), msg, MultiSpan::new(), None);
        self
    }

    /// Prints the span with a warning above it.
    /// This is like [`Diagnostic::warn()`], but it gets its own span.
    #[rustc_lint_diagnostics]
    pub fn span_warn<S: Into<MultiSpan>>(
        &mut self,
        sp: S,
        msg: impl Into<SubdiagnosticMessage>,
    ) -> &mut Self {
        self.sub(Level::Warning(None), msg, sp.into(), None);
        self
    }

    /// Add a help message attached to this diagnostic.
    #[rustc_lint_diagnostics]
    pub fn help(&mut self, msg: impl Into<SubdiagnosticMessage>) -> &mut Self {
        self.sub(Level::Help, msg, MultiSpan::new(), None);
        self
    }

    /// Add a help message attached to this diagnostic with a customizable highlighted message.
    pub fn highlighted_help(&mut self, msg: Vec<(String, Style)>) -> &mut Self {
        self.sub_with_highlights(Level::Help, msg, MultiSpan::new(), None);
        self
    }

    /// Prints the span with some help above it.
    /// This is like [`Diagnostic::help()`], but it gets its own span.
    #[rustc_lint_diagnostics]
    pub fn span_help<S: Into<MultiSpan>>(
        &mut self,
        sp: S,
        msg: impl Into<SubdiagnosticMessage>,
    ) -> &mut Self {
        self.sub(Level::Help, msg, sp.into(), None);
        self
    }

    /// Help the user upgrade to the latest edition.
    /// This is factored out to make sure it does the right thing with `Cargo.toml`.
    pub fn help_use_latest_edition(&mut self) -> &mut Self {
        if std::env::var_os("CARGO").is_some() {
            self.help(&format!("set `edition = \"{}\"` in `Cargo.toml`", LATEST_STABLE_EDITION));
        } else {
            self.help(&format!("pass `--edition {}` to `rustc`", LATEST_STABLE_EDITION));
        }
        self.note("for more on editions, read https://doc.rust-lang.org/edition-guide");
        self
    }

    /// Disallow attaching suggestions this diagnostic.
    /// Any suggestions attached e.g. with the `span_suggestion_*` methods
    /// (before and after the call to `disable_suggestions`) will be ignored.
    pub fn disable_suggestions(&mut self) -> &mut Self {
        self.suggestions = Err(SuggestionsDisabled);
        self
    }

    /// Clear any existing suggestions.
    pub fn clear_suggestions(&mut self) -> &mut Self {
        if let Ok(suggestions) = &mut self.suggestions {
            suggestions.clear();
        }
        self
    }

    /// Helper for pushing to `self.suggestions`, if available (not disable).
    fn push_suggestion(&mut self, suggestion: CodeSuggestion) {
        if let Ok(suggestions) = &mut self.suggestions {
            suggestions.push(suggestion);
        }
    }

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
    }

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
    /// [`Diagnostic::multipart_suggestion()`] but you can set the [`SuggestionStyle`].
    pub fn multipart_suggestion_with_style(
        &mut self,
        msg: impl Into<SubdiagnosticMessage>,
        suggestion: Vec<(Span, String)>,
        applicability: Applicability,
        style: SuggestionStyle,
    ) -> &mut Self {
        assert!(!suggestion.is_empty());
        self.push_suggestion(CodeSuggestion {
            substitutions: vec![Substitution {
                parts: suggestion
                    .into_iter()
                    .map(|(span, snippet)| SubstitutionPart { snippet, span })
                    .collect(),
            }],
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
    }

    /// [`Diagnostic::span_suggestion()`] but you can set the [`SuggestionStyle`].
    pub fn span_suggestion_with_style(
        &mut self,
        sp: Span,
        msg: impl Into<SubdiagnosticMessage>,
        suggestion: impl ToString,
        applicability: Applicability,
        style: SuggestionStyle,
    ) -> &mut Self {
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
    }

    /// Prints out a message with multiple suggested edits of the code.
    /// See also [`Diagnostic::span_suggestion()`].
    pub fn span_suggestions(
        &mut self,
        sp: Span,
        msg: impl Into<SubdiagnosticMessage>,
        suggestions: impl Iterator<Item = String>,
        applicability: Applicability,
    ) -> &mut Self {
        let mut suggestions: Vec<_> = suggestions.collect();
        suggestions.sort();
        let substitutions = suggestions
            .into_iter()
            .map(|snippet| Substitution { parts: vec![SubstitutionPart { snippet, span: sp }] })
            .collect();
        self.push_suggestion(CodeSuggestion {
            substitutions,
            msg: self.subdiagnostic_message_to_diagnostic_message(msg),
            style: SuggestionStyle::ShowCode,
            applicability,
        });
        self
    }

    /// Prints out a message with multiple suggested edits of the code.
    /// See also [`Diagnostic::span_suggestion()`].
    pub fn multipart_suggestions(
        &mut self,
        msg: impl Into<SubdiagnosticMessage>,
        suggestions: impl Iterator<Item = Vec<(Span, String)>>,
        applicability: Applicability,
    ) -> &mut Self {
        self.push_suggestion(CodeSuggestion {
            substitutions: suggestions
                .map(|sugg| Substitution {
                    parts: sugg
                        .into_iter()
                        .map(|(span, snippet)| SubstitutionPart { snippet, span })
                        .collect(),
                })
                .collect(),
            msg: self.subdiagnostic_message_to_diagnostic_message(msg),
            style: SuggestionStyle::ShowCode,
            applicability,
        });
        self
    }
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
    }

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

    /// Adds a suggestion to the JSON output that will not be shown in the CLI.
    ///
    /// This is intended to be used for suggestions that are *very* obvious in what the changes
    /// need to be from the message, but we still want other tools to be able to apply them.
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
    }

    /// Add a subdiagnostic from a type that implements `SessionSubdiagnostic` - see
    /// [rustc_macros::SessionSubdiagnostic].
    pub fn subdiagnostic(&mut self, subdiagnostic: impl AddSubdiagnostic) -> &mut Self {
        subdiagnostic.add_to_diagnostic(self);
        self
    }

    pub fn set_span<S: Into<MultiSpan>>(&mut self, sp: S) -> &mut Self {
        self.span = sp.into();
        if let Some(span) = self.span.primary_span() {
            self.sort_span = span;
        }
        self
    }

    pub fn set_is_lint(&mut self) -> &mut Self {
        self.is_lint = true;
        self
    }

    pub fn code(&mut self, s: DiagnosticId) -> &mut Self {
        self.code = Some(s);
        self
    }

    pub fn clear_code(&mut self) -> &mut Self {
        self.code = None;
        self
    }

    pub fn get_code(&self) -> Option<DiagnosticId> {
        self.code.clone()
    }

    pub fn set_primary_message(&mut self, msg: impl Into<DiagnosticMessage>) -> &mut Self {
        self.message[0] = (msg.into(), Style::NoStyle);
        self
    }

    pub fn args(&self) -> &[DiagnosticArg<'static>] {
        &self.args
    }

    pub fn set_arg(
        &mut self,
        name: impl Into<Cow<'static, str>>,
        arg: impl IntoDiagnosticArg,
    ) -> &mut Self {
        self.args.push((name.into(), arg.into_diagnostic_arg()));
        self
    }

    pub fn styled_message(&self) -> &[(DiagnosticMessage, Style)] {
        &self.message
    }

    /// Helper function that takes a `SubdiagnosticMessage` and returns a `DiagnosticMessage` by
    /// combining it with the primary message of the diagnostic (if translatable, otherwise it just
    /// passes the user's string along).
    fn subdiagnostic_message_to_diagnostic_message(
        &self,
        attr: impl Into<SubdiagnosticMessage>,
    ) -> DiagnosticMessage {
        let msg =
            self.message.iter().map(|(msg, _)| msg).next().expect("diagnostic with no messages");
        msg.with_subdiagnostic_message(attr.into())
    }

    /// Convenience function for internal use, clients should use one of the
    /// public methods above.
    ///
    /// Used by `proc_macro_server` for implementing `server::Diagnostic`.
    pub fn sub(
        &mut self,
        level: Level,
        message: impl Into<SubdiagnosticMessage>,
        span: MultiSpan,
        render_span: Option<MultiSpan>,
    ) {
        let sub = SubDiagnostic {
            level,
            message: vec![(
                self.subdiagnostic_message_to_diagnostic_message(message),
                Style::NoStyle,
            )],
            span,
            render_span,
        };
        self.children.push(sub);
    }

    /// Convenience function for internal use, clients should use one of the
    /// public methods above.
    fn sub_with_highlights<M: Into<SubdiagnosticMessage>>(
        &mut self,
        level: Level,
        message: Vec<(M, Style)>,
        span: MultiSpan,
        render_span: Option<MultiSpan>,
    ) {
        let message = message
            .into_iter()
            .map(|m| (self.subdiagnostic_message_to_diagnostic_message(m.0), m.1))
            .collect();
        let sub = SubDiagnostic { level, message, span, render_span };
        self.children.push(sub);
    }

    /// Fields used for Hash, and PartialEq trait
    fn keys(
        &self,
    ) -> (
        &Level,
        &[(DiagnosticMessage, Style)],
        &Option<DiagnosticId>,
        &MultiSpan,
        &Result<Vec<CodeSuggestion>, SuggestionsDisabled>,
        Option<&[SubDiagnostic]>,
    ) {
        (
            &self.level,
            &self.message,
            &self.code,
            &self.span,
            &self.suggestions,
            (if self.is_lint { None } else { Some(&self.children) }),
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
