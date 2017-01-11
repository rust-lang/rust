// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use CodeSuggestion;
use Level;
use RenderSpan;
use RenderSpan::Suggestion;
use std::fmt;
use syntax_pos::{MultiSpan, Span};
use snippet::Style;

#[must_use]
#[derive(Clone, Debug, PartialEq)]
pub struct Diagnostic {
    pub level: Level,
    pub message: Vec<(String, Style)>,
    pub code: Option<String>,
    pub span: MultiSpan,
    pub children: Vec<SubDiagnostic>,
}

/// For example a note attached to an error.
#[derive(Clone, Debug, PartialEq)]
pub struct SubDiagnostic {
    pub level: Level,
    pub message: Vec<(String, Style)>,
    pub span: MultiSpan,
    pub render_span: Option<RenderSpan>,
}

impl Diagnostic {
    pub fn new(level: Level, message: &str) -> Self {
        Diagnostic::new_with_code(level, None, message)
    }

    pub fn new_with_code(level: Level, code: Option<String>, message: &str) -> Self {
        Diagnostic {
            level: level,
            message: vec![(message.to_owned(), Style::NoStyle)],
            code: code,
            span: MultiSpan::new(),
            children: vec![],
        }
    }

    /// Cancel the diagnostic (a structured diagnostic must either be emitted or
    /// cancelled or it will panic when dropped).
    /// BEWARE: if this DiagnosticBuilder is an error, then creating it will
    /// bump the error count on the Handler and cancelling it won't undo that.
    /// If you want to decrement the error count you should use `Handler::cancel`.
    pub fn cancel(&mut self) {
        self.level = Level::Cancelled;
    }

    pub fn cancelled(&self) -> bool {
        self.level == Level::Cancelled
    }

    pub fn is_fatal(&self) -> bool {
        self.level == Level::Fatal
    }

    /// Add a span/label to be included in the resulting snippet.
    /// This is pushed onto the `MultiSpan` that was created when the
    /// diagnostic was first built. If you don't call this function at
    /// all, and you just supplied a `Span` to create the diagnostic,
    /// then the snippet will just include that `Span`, which is
    /// called the primary span.
    pub fn span_label(&mut self, span: Span, label: &fmt::Display)
                      -> &mut Self {
        self.span.push_span_label(span, format!("{}", label));
        self
    }

    pub fn note_expected_found(&mut self,
                               label: &fmt::Display,
                               expected: &fmt::Display,
                               found: &fmt::Display)
                               -> &mut Self
    {
        self.note_expected_found_extra(label, expected, found, &"", &"")
    }

    pub fn note_expected_found_extra(&mut self,
                                     label: &fmt::Display,
                                     expected: &fmt::Display,
                                     found: &fmt::Display,
                                     expected_extra: &fmt::Display,
                                     found_extra: &fmt::Display)
                                     -> &mut Self
    {
        // For now, just attach these as notes
        self.highlighted_note(vec![
            (format!("expected {} `", label), Style::NoStyle),
            (format!("{}", expected), Style::Highlight),
            (format!("`{}\n", expected_extra), Style::NoStyle),
            (format!("   found {} `", label), Style::NoStyle),
            (format!("{}", found), Style::Highlight),
            (format!("`{}", found_extra), Style::NoStyle),
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

    pub fn span_note<S: Into<MultiSpan>>(&mut self,
                                         sp: S,
                                         msg: &str)
                                         -> &mut Self {
        self.sub(Level::Note, msg, sp.into(), None);
        self
    }

    pub fn warn(&mut self, msg: &str) -> &mut Self {
        self.sub(Level::Warning, msg, MultiSpan::new(), None);
        self
    }

    pub fn span_warn<S: Into<MultiSpan>>(&mut self,
                                         sp: S,
                                         msg: &str)
                                         -> &mut Self {
        self.sub(Level::Warning, msg, sp.into(), None);
        self
    }

    pub fn help(&mut self , msg: &str) -> &mut Self {
        self.sub(Level::Help, msg, MultiSpan::new(), None);
        self
    }

    pub fn span_help<S: Into<MultiSpan>>(&mut self,
                                         sp: S,
                                         msg: &str)
                                         -> &mut Self {
        self.sub(Level::Help, msg, sp.into(), None);
        self
    }

    /// Prints out a message with a suggested edit of the code.
    ///
    /// See `diagnostic::RenderSpan::Suggestion` for more information.
    pub fn span_suggestion<S: Into<MultiSpan>>(&mut self,
                                               sp: S,
                                               msg: &str,
                                               suggestion: String)
                                               -> &mut Self {
        self.sub(Level::Help,
                 msg,
                 MultiSpan::new(),
                 Some(Suggestion(CodeSuggestion {
                     msp: sp.into(),
                     substitutes: vec![suggestion],
                 })));
        self
    }

    pub fn set_span<S: Into<MultiSpan>>(&mut self, sp: S) -> &mut Self {
        self.span = sp.into();
        self
    }

    pub fn code(&mut self, s: String) -> &mut Self {
        self.code = Some(s);
        self
    }

    pub fn message(&self) -> String {
        self.message.iter().map(|i| i.0.to_owned()).collect::<String>()
    }

    pub fn styled_message(&self) -> &Vec<(String, Style)> {
        &self.message
    }

    pub fn level(&self) -> Level {
        self.level
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
    fn sub(&mut self,
           level: Level,
           message: &str,
           span: MultiSpan,
           render_span: Option<RenderSpan>) {
        let sub = SubDiagnostic {
            level: level,
            message: vec![(message.to_owned(), Style::NoStyle)],
            span: span,
            render_span: render_span,
        };
        self.children.push(sub);
    }

    /// Convenience function for internal use, clients should use one of the
    /// public methods above.
    fn sub_with_highlights(&mut self,
                           level: Level,
                           message: Vec<(String, Style)>,
                           span: MultiSpan,
                           render_span: Option<RenderSpan>) {
        let sub = SubDiagnostic {
            level: level,
            message: message,
            span: span,
            render_span: render_span,
        };
        self.children.push(sub);
    }
}

impl SubDiagnostic {
    pub fn message(&self) -> String {
        self.message.iter().map(|i| i.0.to_owned()).collect::<String>()
    }

    pub fn styled_message(&self) -> &Vec<(String, Style)> {
        &self.message
    }
}
