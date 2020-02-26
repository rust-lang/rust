use crate::snippet::Style;
use crate::Applicability;
use crate::CodeSuggestion;
use crate::Level;
use crate::Substitution;
use crate::SubstitutionPart;
use crate::SuggestionStyle;
use rustc_span::{MultiSpan, Span, DUMMY_SP};
use std::fmt;

#[must_use]
#[derive(Clone, Debug, PartialEq, Hash, RustcEncodable, RustcDecodable)]
pub struct Diagnostic {
    pub level: Level,
    pub message: Vec<(String, Style)>,
    pub code: Option<DiagnosticId>,
    pub span: MultiSpan,
    pub children: Vec<SubDiagnostic>,
    pub suggestions: Vec<CodeSuggestion>,

    /// This is not used for highlighting or rendering any error message.  Rather, it can be used
    /// as a sort key to sort a buffer of diagnostics.  By default, it is the primary span of
    /// `span` if there is one.  Otherwise, it is `DUMMY_SP`.
    pub sort_span: Span,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub enum DiagnosticId {
    Error(String),
    Lint(String),
}

/// For example a note attached to an error.
#[derive(Clone, Debug, PartialEq, Hash, RustcEncodable, RustcDecodable)]
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
            suggestions: vec![],
            sort_span: DUMMY_SP,
        }
    }

    pub fn is_error(&self) -> bool {
        match self.level {
            Level::Bug | Level::Fatal | Level::Error | Level::FailureNote => true,

            Level::Warning | Level::Note | Level::Help | Level::Cancelled => false,
        }
    }

    /// Cancel the diagnostic (a structured diagnostic must either be emitted or
    /// canceled or it will panic when dropped).
    pub fn cancel(&mut self) {
        self.level = Level::Cancelled;
    }

    pub fn cancelled(&self) -> bool {
        self.level == Level::Cancelled
    }

    /// Set the sorting span.
    pub fn set_sort_span(&mut self, sp: Span) {
        self.sort_span = sp;
    }

    /// Adds a span/label to be included in the resulting snippet.
    /// This label will be shown together with the original span/label used when creating the
    /// diagnostic, *not* a span added by one of the `span_*` methods.
    ///
    /// This is pushed onto the `MultiSpan` that was created when the
    /// diagnostic was first built. If you don't call this function at
    /// all, and you just supplied a `Span` to create the diagnostic,
    /// then the snippet will just include that `Span`, which is
    /// called the primary span.
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

    pub fn note_expected_found(
        &mut self,
        expected_label: &dyn fmt::Display,
        expected: DiagnosticStyledString,
        found_label: &dyn fmt::Display,
        found: DiagnosticStyledString,
    ) -> &mut Self {
        self.note_expected_found_extra(expected_label, expected, found_label, found, &"", &"")
    }

    pub fn note_unsuccessfull_coercion(
        &mut self,
        expected: DiagnosticStyledString,
        found: DiagnosticStyledString,
    ) -> &mut Self {
        let mut msg: Vec<_> =
            vec![(format!("required when trying to coerce from type `"), Style::NoStyle)];
        msg.extend(expected.0.iter().map(|x| match *x {
            StringPart::Normal(ref s) => (s.to_owned(), Style::NoStyle),
            StringPart::Highlighted(ref s) => (s.to_owned(), Style::Highlight),
        }));
        msg.push((format!("` to type '"), Style::NoStyle));
        msg.extend(found.0.iter().map(|x| match *x {
            StringPart::Normal(ref s) => (s.to_owned(), Style::NoStyle),
            StringPart::Highlighted(ref s) => (s.to_owned(), Style::Highlight),
        }));
        msg.push((format!("`"), Style::NoStyle));

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
        let expected_label = format!("expected {}", expected_label);

        let found_label = format!("found {}", found_label);
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

    pub fn note(&mut self, msg: &str) -> &mut Self {
        self.sub(Level::Note, msg, MultiSpan::new(), None);
        self
    }

    pub fn highlighted_note(&mut self, msg: Vec<(String, Style)>) -> &mut Self {
        self.sub_with_highlights(Level::Note, msg, MultiSpan::new(), None);
        self
    }

    /// Prints the span with a note above it.
    pub fn span_note<S: Into<MultiSpan>>(&mut self, sp: S, msg: &str) -> &mut Self {
        self.sub(Level::Note, msg, sp.into(), None);
        self
    }

    pub fn warn(&mut self, msg: &str) -> &mut Self {
        self.sub(Level::Warning, msg, MultiSpan::new(), None);
        self
    }

    /// Prints the span with a warn above it.
    pub fn span_warn<S: Into<MultiSpan>>(&mut self, sp: S, msg: &str) -> &mut Self {
        self.sub(Level::Warning, msg, sp.into(), None);
        self
    }

    pub fn help(&mut self, msg: &str) -> &mut Self {
        self.sub(Level::Help, msg, MultiSpan::new(), None);
        self
    }

    /// Prints the span with some help above it.
    pub fn span_help<S: Into<MultiSpan>>(&mut self, sp: S, msg: &str) -> &mut Self {
        self.sub(Level::Help, msg, sp.into(), None);
        self
    }

    pub fn multipart_suggestion(
        &mut self,
        msg: &str,
        suggestion: Vec<(Span, String)>,
        applicability: Applicability,
    ) -> &mut Self {
        self.suggestions.push(CodeSuggestion {
            substitutions: vec![Substitution {
                parts: suggestion
                    .into_iter()
                    .map(|(span, snippet)| SubstitutionPart { snippet, span })
                    .collect(),
            }],
            msg: msg.to_owned(),
            style: SuggestionStyle::ShowCode,
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
        msg: &str,
        suggestion: Vec<(Span, String)>,
        applicability: Applicability,
    ) -> &mut Self {
        self.suggestions.push(CodeSuggestion {
            substitutions: vec![Substitution {
                parts: suggestion
                    .into_iter()
                    .map(|(span, snippet)| SubstitutionPart { snippet, span })
                    .collect(),
            }],
            msg: msg.to_owned(),
            style: SuggestionStyle::CompletelyHidden,
            applicability,
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

    pub fn span_suggestion_with_style(
        &mut self,
        sp: Span,
        msg: &str,
        suggestion: String,
        applicability: Applicability,
        style: SuggestionStyle,
    ) -> &mut Self {
        self.suggestions.push(CodeSuggestion {
            substitutions: vec![Substitution {
                parts: vec![SubstitutionPart { snippet: suggestion, span: sp }],
            }],
            msg: msg.to_owned(),
            style,
            applicability,
        });
        self
    }

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
    pub fn span_suggestions(
        &mut self,
        sp: Span,
        msg: &str,
        suggestions: impl Iterator<Item = String>,
        applicability: Applicability,
    ) -> &mut Self {
        self.suggestions.push(CodeSuggestion {
            substitutions: suggestions
                .map(|snippet| Substitution { parts: vec![SubstitutionPart { snippet, span: sp }] })
                .collect(),
            msg: msg.to_owned(),
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

    /// Prints out a message with for a suggestion without showing the suggested code.
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

    /// Adds a suggestion to the json output, but otherwise remains silent/undisplayed in the cli.
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

    pub fn set_span<S: Into<MultiSpan>>(&mut self, sp: S) -> &mut Self {
        self.span = sp.into();
        if let Some(span) = self.span.primary_span() {
            self.sort_span = span;
        }
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

    pub fn set_primary_message<M: Into<String>>(&mut self, msg: M) -> &mut Self {
        self.message[0] = (msg.into(), Style::NoStyle);
        self
    }

    pub fn message(&self) -> String {
        self.message.iter().map(|i| i.0.as_str()).collect::<String>()
    }

    pub fn styled_message(&self) -> &Vec<(String, Style)> {
        &self.message
    }

    /// Used by a lint. Copies over all details *but* the "main
    /// message".
    pub fn copy_details_not_message(&mut self, from: &Diagnostic) {
        self.span = from.span.clone();
        self.code = from.code.clone();
        self.children.extend(from.children.iter().cloned())
    }

    /// Convenience function for internal use, clients should use one of the
    /// public methods above.
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
}

impl SubDiagnostic {
    pub fn message(&self) -> String {
        self.message.iter().map(|i| i.0.as_str()).collect::<String>()
    }

    pub fn styled_message(&self) -> &Vec<(String, Style)> {
        &self.message
    }
}
