// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use Span;

use rustc_errors as errors;
use syntax_pos::MultiSpan;

/// An enum representing a diagnostic level.
#[unstable(feature = "proc_macro_diagnostic", issue = "38356")]
#[derive(Copy, Clone, Debug)]
#[non_exhaustive]
pub enum Level {
    /// An error.
    Error,
    /// A warning.
    Warning,
    /// A note.
    Note,
    /// A help message.
    Help,
}

/// A structure representing a diagnostic message and associated children
/// messages.
#[unstable(feature = "proc_macro_diagnostic", issue = "38356")]
#[derive(Clone, Debug)]
pub struct Diagnostic {
    level: Level,
    message: String,
    span: Option<Span>,
    children: Vec<Diagnostic>
}

macro_rules! diagnostic_child_methods {
    ($spanned:ident, $regular:ident, $level:expr) => (
        /// Add a new child diagnostic message to `self` with the level
        /// identified by this methods name with the given `span` and `message`.
        #[unstable(feature = "proc_macro_diagnostic", issue = "38356")]
        pub fn $spanned<T: Into<String>>(mut self, span: Span, message: T) -> Diagnostic {
            self.children.push(Diagnostic::spanned(span, $level, message));
            self
        }

        /// Add a new child diagnostic message to `self` with the level
        /// identified by this method's name with the given `message`.
        #[unstable(feature = "proc_macro_diagnostic", issue = "38356")]
        pub fn $regular<T: Into<String>>(mut self, message: T) -> Diagnostic {
            self.children.push(Diagnostic::new($level, message));
            self
        }
    )
}

#[derive(Debug, Clone)]
#[unstable(feature = "proc_macro_diagnostic", issue = "38356")]
/// Iterator over the children diagnostics of a `Diagnostic`.
pub struct Children<'a>(::std::slice::Iter<'a, Diagnostic>);

#[unstable(feature = "proc_macro_diagnostic", issue = "38356")]
impl<'a> Iterator for Children<'a> {
    type Item = &'a Diagnostic;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

#[unstable(feature = "proc_macro_diagnostic", issue = "38356")]
impl Diagnostic {
    /// Create a new diagnostic with the given `level` and `message`.
    #[unstable(feature = "proc_macro_diagnostic", issue = "38356")]
    pub fn new<T: Into<String>>(level: Level, message: T) -> Diagnostic {
        Diagnostic {
            level: level,
            message: message.into(),
            span: None,
            children: vec![]
        }
    }

    /// Create a new diagnostic with the given `level` and `message` pointing to
    /// the given `span`.
    #[unstable(feature = "proc_macro_diagnostic", issue = "38356")]
    pub fn spanned<T: Into<String>>(span: Span, level: Level, message: T) -> Diagnostic {
        Diagnostic {
            level: level,
            message: message.into(),
            span: Some(span),
            children: vec![]
        }
    }

    diagnostic_child_methods!(span_error, error, Level::Error);
    diagnostic_child_methods!(span_warning, warning, Level::Warning);
    diagnostic_child_methods!(span_note, note, Level::Note);
    diagnostic_child_methods!(span_help, help, Level::Help);

    /// Returns the diagnostic `level` for `self`.
    #[unstable(feature = "proc_macro_diagnostic", issue = "38356")]
    pub fn level(&self) -> Level {
        self.level
    }

    /// Sets the level in `self` to `level`.
    #[unstable(feature = "proc_macro_diagnostic", issue = "38356")]
    pub fn set_level(&mut self, level: Level) {
        self.level = level;
    }

    /// Returns the message in `self`.
    #[unstable(feature = "proc_macro_diagnostic", issue = "38356")]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Sets the message in `self` to `message`.
    #[unstable(feature = "proc_macro_diagnostic", issue = "38356")]
    pub fn set_message<T: Into<String>>(&mut self, message: T) {
        self.message = message.into();
    }

    /// Returns the `Span` in `self`.
    #[unstable(feature = "proc_macro_diagnostic", issue = "38356")]
    pub fn span(&self) -> Option<Span> {
        self.span
    }

    /// Sets the `Span` in `self` to `span`.
    #[unstable(feature = "proc_macro_diagnostic", issue = "38356")]
    pub fn set_span(&mut self, span: Span) {
        self.span = Some(span);
    }

    /// Returns an iterator over the children diagnostics of `self`.
    #[unstable(feature = "proc_macro_diagnostic", issue = "38356")]
    pub fn children(&self) -> Children {
        Children(self.children.iter())
    }

    /// Emit the diagnostic.
    #[unstable(feature = "proc_macro_diagnostic", issue = "38356")]
    pub fn emit(self) {
        let level = self.level.to_internal();
        let mut diag = errors::Diagnostic::new(level, &*self.message);

        if let Some(span) = self.span {
            diag.set_span(span.0);
        }

        for child in self.children {
            let span = child.span.map_or(MultiSpan::new(), |s| s.0.into());
            let level = child.level.to_internal();
            diag.sub(level, &*child.message, span, None);
        }

        ::__internal::with_sess(move |sess, _| {
            errors::DiagnosticBuilder::new_diagnostic(&sess.span_diagnostic, diag).emit();
        });
    }
}
