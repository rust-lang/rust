use crate::snippet::Style;
use crate::CodeSuggestion;
use crate::Level;
use crate::Substitution;
use crate::SubstitutionPart;
use crate::SuggestionStyle;
use crate::ToolMetadata;
use rustc_lint_defs::Applicability;
use rustc_serialize::json::Json;
use rustc_span::{MultiSpan, Span, DUMMY_SP};
use std::fmt;
use std::hash::{Hash, Hasher};

/// Error type for `Diagnostic`'s `suggestions` field, indicating that
/// `.disable_suggestions()` was called on the `Diagnostic`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Encodable, Decodable)]
pub struct SuggestionsDisabled;

#[must_use]
#[derive(Clone, Debug, Encodable, Decodable)]
pub struct Diagnostic {
    pub level: Level,
    pub message: Vec<(String, Style)>,
    pub code: Option<DiagnosticId>,
    pub span: MultiSpan,
    pub children: Vec<SubDiagnostic>,
    pub suggestions: Result<Vec<CodeSuggestion>, SuggestionsDisabled>,

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
    pub message: Vec<(String, Style)>,
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
    pub fn new(level: Level, message: &str) -> Self {
        Diagnostic::new_with_code(level, None, message)
    }

    pub fn new_with_code(level: Level, code: Option<DiagnosticId>, message: &str) -> Self {
        Diagnostic {
            level,
            message: vec![(message.to_owned(), Style::NoStyle)],
            code,
            span: MultiSpan::new(),
            children: vec![],
            suggestions: Ok(vec![]),
            sort_span: DUMMY_SP,
            is_lint: false,
        }
    }

    pub fn is_error(&self) -> bool {
        match self.level {
            Level::Bug | Level::Fatal | Level::Error { .. } | Level::FailureNote => true,

            Level::Warning | Level::Note | Level::Help | Level::Cancelled | Level::Allow => false,
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

    /// Cancel the diagnostic (a structured diagnostic must either be emitted or
    /// canceled or it will panic when dropped).
    pub fn cancel(&mut self) {
        self.level = Level::Cancelled;
    }

    /// Check if this diagnostic [was cancelled][Self::cancel()].
    pub fn cancelled(&self) -> bool {
        self.level == Level::Cancelled
    }

    /// Adds a span/label to be included in the resulting snippet.
    ///
    /// This is pushed onto the [`MultiSpan`] that was created when the diagnostic
    /// was first built. That means it will be shown together with the original
    /// span/label, *not* a span added by one of the `span_{note,warn,help,suggestions}` methods.
    ///
    /// This span is *not* considered a ["primary span"][`MultiSpan`]; only
    /// the `Span` supplied when creating the diagnostic is primary.
    pub fn span_label<T: Into<String>>(&mut self, span: Span, label: T) -> &mut Self {
        self.span.push_span_label(span, label.into());
        self
    }

    pub fn replace_span_with(&mut self, after: Span) -> &mut Self {
        let before = self.span.clone();
        self.set_span(after);
        for span_label in before.span_labels() {
            if let Some(label) = span_label.label {
                self.span_label(after, label);
            }
        }
        self
    }

    crate fn note_expected_found(
        &mut self,
        expected_label: &dyn fmt::Display,
        expected: DiagnosticStyledString,
        found_label: &dyn fmt::Display,
        found: DiagnosticStyledString,
    ) -> &mut Self {
        self.note_expected_found_extra(expected_label, expected, found_label, found, &"", &"")
    }

    crate fn note_unsuccessful_coercion(
        &mut self,
        expected: DiagnosticStyledString,
        found: DiagnosticStyledString,
    ) -> &mut Self {
        let mut msg: Vec<_> =
            vec![("required when trying to coerce from type `".to_string(), Style::NoStyle)];
        msg.extend(expected.0.iter().map(|x| match *x {
            StringPart::Normal(ref s) => (s.to_owned(), Style::NoStyle),
            StringPart::Highlighted(ref s) => (s.to_owned(), Style::Highlight),
        }));
        msg.push(("` to type '".to_string(), Style::NoStyle));
        msg.extend(found.0.iter().map(|x| match *x {
            StringPart::Normal(ref s) => (s.to_owned(), Style::NoStyle),
            StringPart::Highlighted(ref s) => (s.to_owned(), Style::Highlight),
        }));
        msg.push(("`".to_string(), Style::NoStyle));

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

    pub fn note_trait_signature(&mut self, name: String, signature: String) -> &mut Self {
        self.highlighted_note(vec![
            (format!("`{}` from trait: `", name), Style::NoStyle),
            (signature, Style::Highlight),
            ("`".to_string(), Style::NoStyle),
        ]);
        self
    }

    /// Add a note attached to this diagnostic.
    pub fn note(&mut self, msg: &str) -> &mut Self {
        self.sub(Level::Note, msg, MultiSpan::new(), None);
        self
    }

    pub fn highlighted_note(&mut self, msg: Vec<(String, Style)>) -> &mut Self {
        self.sub_with_highlights(Level::Note, msg, MultiSpan::new(), None);
        self
    }

    /// Prints the span with a note above it.
    /// This is like [`Diagnostic::note()`], but it gets its own span.
    crate fn span_note<S: Into<MultiSpan>>(&mut self, sp: S, msg: &str) -> &mut Self {
        self.sub(Level::Note, msg, sp.into(), None);
        self
    }

    /// Add a warning attached to this diagnostic.
    crate fn warn(&mut self, msg: &str) -> &mut Self {
        self.sub(Level::Warning, msg, MultiSpan::new(), None);
        self
    }

    /// Prints the span with a warning above it.
    /// This is like [`Diagnostic::warn()`], but it gets its own span.
    crate fn span_warn<S: Into<MultiSpan>>(&mut self, sp: S, msg: &str) -> &mut Self {
        self.sub(Level::Warning, msg, sp.into(), None);
        self
    }

    /// Add a help message attached to this diagnostic.
    crate fn help(&mut self, msg: &str) -> &mut Self {
        self.sub(Level::Help, msg, MultiSpan::new(), None);
        self
    }

    /// Prints the span with some help above it.
    /// This is like [`Diagnostic::help()`], but it gets its own span.
    crate fn span_help<S: Into<MultiSpan>>(&mut self, sp: S, msg: &str) -> &mut Self {
        self.sub(Level::Help, msg, sp.into(), None);
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
        if let Ok(suggestions) = &mut self.suggestions {
            suggestions.push(suggestion);
        }
    }

    /// Show a suggestion that has multiple parts to it.
    /// In other words, multiple changes need to be applied as part of this suggestion.
    pub fn multipart_suggestion(
        &mut self,
        msg: &str,
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
        msg: &str,
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
        msg: &str,
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
            msg: msg.to_owned(),
            style,
            applicability,
            tool_metadata: Default::default(),
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
        msg: &str,
        suggestion: Vec<(Span, String)>,
        applicability: Applicability,
    ) -> &mut Self {
        assert!(!suggestion.is_empty());
        self.push_suggestion(CodeSuggestion {
            substitutions: vec![Substitution {
                parts: suggestion
                    .into_iter()
                    .map(|(span, snippet)| SubstitutionPart { snippet, span })
                    .collect(),
            }],
            msg: msg.to_owned(),
            style: SuggestionStyle::CompletelyHidden,
            applicability,
            tool_metadata: Default::default(),
        });
        self
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
        msg: &str,
        suggestion: String,
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
        msg: &str,
        suggestion: String,
        applicability: Applicability,
        style: SuggestionStyle,
    ) -> &mut Self {
        self.push_suggestion(CodeSuggestion {
            substitutions: vec![Substitution {
                parts: vec![SubstitutionPart { snippet: suggestion, span: sp }],
            }],
            msg: msg.to_owned(),
            style,
            applicability,
            tool_metadata: Default::default(),
        });
        self
    }

    /// Always show the suggested change.
    pub fn span_suggestion_verbose(
        &mut self,
        sp: Span,
        msg: &str,
        suggestion: String,
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
        msg: &str,
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
            msg: msg.to_owned(),
            style: SuggestionStyle::ShowCode,
            applicability,
            tool_metadata: Default::default(),
        });
        self
    }

    /// Prints out a message with multiple suggested edits of the code.
    /// See also [`Diagnostic::span_suggestion()`].
    pub fn multipart_suggestions(
        &mut self,
        msg: &str,
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
            msg: msg.to_owned(),
            style: SuggestionStyle::ShowCode,
            applicability,
            tool_metadata: Default::default(),
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
        msg: &str,
        suggestion: String,
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
        msg: &str,
        suggestion: String,
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
        msg: &str,
        suggestion: String,
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

    /// Adds a suggestion intended only for a tool. The intent is that the metadata encodes
    /// the suggestion in a tool-specific way, as it may not even directly involve Rust code.
    pub fn tool_only_suggestion_with_metadata(
        &mut self,
        msg: &str,
        applicability: Applicability,
        tool_metadata: Json,
    ) {
        self.push_suggestion(CodeSuggestion {
            substitutions: vec![],
            msg: msg.to_owned(),
            style: SuggestionStyle::CompletelyHidden,
            applicability,
            tool_metadata: ToolMetadata::new(tool_metadata),
        })
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

    crate fn set_primary_message<M: Into<String>>(&mut self, msg: M) -> &mut Self {
        self.message[0] = (msg.into(), Style::NoStyle);
        self
    }

    pub fn message(&self) -> String {
        self.message.iter().map(|i| i.0.as_str()).collect::<String>()
    }

    pub fn styled_message(&self) -> &Vec<(String, Style)> {
        &self.message
    }

    /// Convenience function for internal use, clients should use one of the
    /// public methods above.
    ///
    /// Used by `proc_macro_server` for implementing `server::Diagnostic`.
    pub fn sub(
        &mut self,
        level: Level,
        message: &str,
        span: MultiSpan,
        render_span: Option<MultiSpan>,
    ) {
        let sub = SubDiagnostic {
            level,
            message: vec![(message.to_owned(), Style::NoStyle)],
            span,
            render_span,
        };
        self.children.push(sub);
    }

    /// Convenience function for internal use, clients should use one of the
    /// public methods above.
    fn sub_with_highlights(
        &mut self,
        level: Level,
        message: Vec<(String, Style)>,
        span: MultiSpan,
        render_span: Option<MultiSpan>,
    ) {
        let sub = SubDiagnostic { level, message, span, render_span };
        self.children.push(sub);
    }

    /// Fields used for Hash, and PartialEq trait
    fn keys(
        &self,
    ) -> (
        &Level,
        &Vec<(String, Style)>,
        &Option<DiagnosticId>,
        &MultiSpan,
        &Result<Vec<CodeSuggestion>, SuggestionsDisabled>,
        Option<&Vec<SubDiagnostic>>,
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

impl SubDiagnostic {
    pub fn message(&self) -> String {
        self.message.iter().map(|i| i.0.as_str()).collect::<String>()
    }

    pub fn styled_message(&self) -> &Vec<(String, Style)> {
        &self.message
    }
}
