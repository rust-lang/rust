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

use rustc_errors as rustc;

/// An enum representing a diagnostic level.
#[unstable(feature = "proc_macro", issue = "38356")]
#[derive(Copy, Clone, Debug)]
pub enum Level {
    /// An error.
    Error,
    /// A warning.
    Warning,
    /// A note.
    Note,
    /// A help message.
    Help,
    #[doc(hidden)]
    __Nonexhaustive,
}

/// A structure representing a diagnostic message and associated children
/// messages.
#[unstable(feature = "proc_macro", issue = "38356")]
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
        #[unstable(feature = "proc_macro", issue = "38356")]
        pub fn $spanned<T: Into<String>>(mut self, span: Span, message: T) -> Diagnostic {
            self.children.push(Diagnostic::spanned(span, $level, message));
            self
        }

        /// Add a new child diagnostic message to `self` with the level
        /// identified by this method's name with the given `message`.
        #[unstable(feature = "proc_macro", issue = "38356")]
        pub fn $regular<T: Into<String>>(mut self, message: T) -> Diagnostic {
            self.children.push(Diagnostic::new($level, message));
            self
        }
    )
}

impl Diagnostic {
    /// Create a new diagnostic with the given `level` and `message`.
    #[unstable(feature = "proc_macro", issue = "38356")]
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
    #[unstable(feature = "proc_macro", issue = "38356")]
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
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn level(&self) -> Level {
        self.level
    }

    /// Emit the diagnostic.
    #[unstable(feature = "proc_macro", issue = "38356")]
    pub fn emit(self) {
        ::__internal::with_sess(move |(sess, _)| {
            let handler = &sess.span_diagnostic;
            let level = __internal::level_to_internal_level(self.level);
            let mut diag = rustc::DiagnosticBuilder::new(handler, level, &*self.message);

            if let Some(span) = self.span {
                diag.set_span(span.0);
            }

            for child in self.children {
                let span = child.span.map(|s| s.0);
                let level = __internal::level_to_internal_level(child.level);
                diag.sub(level, &*child.message, span);
            }

            diag.emit();
        });
    }
}

#[unstable(feature = "proc_macro_internals", issue = "27812")]
#[doc(hidden)]
pub mod __internal {
    use super::{Level, rustc};

    pub fn level_to_internal_level(level: Level) -> rustc::Level {
        match level {
            Level::Error => rustc::Level::Error,
            Level::Warning => rustc::Level::Warning,
            Level::Note => rustc::Level::Note,
            Level::Help => rustc::Level::Help,
            Level::__Nonexhaustive => unreachable!("Level::__Nonexhaustive")
        }
    }
}
