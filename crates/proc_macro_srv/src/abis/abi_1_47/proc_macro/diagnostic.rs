//! lib-proc-macro diagnostic
//!
//! Copy from <https://github.com/rust-lang/rust/blob/6050e523bae6de61de4e060facc43dc512adaccd/src/libproc_macro/diagnostic.rs>
//! augmented with removing unstable features

use super::Span;

/// An enum representing a diagnostic level.
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

/// Trait implemented by types that can be converted into a set of `Span`s.
pub trait MultiSpan {
    /// Converts `self` into a `Vec<Span>`.
    fn into_spans(self) -> Vec<Span>;
}

impl MultiSpan for Span {
    fn into_spans(self) -> Vec<Span> {
        vec![self]
    }
}

impl MultiSpan for Vec<Span> {
    fn into_spans(self) -> Vec<Span> {
        self
    }
}

impl<'a> MultiSpan for &'a [Span] {
    fn into_spans(self) -> Vec<Span> {
        self.to_vec()
    }
}

/// A structure representing a diagnostic message and associated children
/// messages.
#[derive(Clone, Debug)]
pub struct Diagnostic {
    level: Level,
    message: String,
    spans: Vec<Span>,
    children: Vec<Diagnostic>,
}

macro_rules! diagnostic_child_methods {
    ($spanned:ident, $regular:ident, $level:expr) => {
        /// Adds a new child diagnostic message to `self` with the level
        /// identified by this method's name with the given `spans` and
        /// `message`.
        pub fn $spanned<S, T>(mut self, spans: S, message: T) -> Diagnostic
        where
            S: MultiSpan,
            T: Into<String>,
        {
            self.children.push(Diagnostic::spanned(spans, $level, message));
            self
        }

        /// Adds a new child diagnostic message to `self` with the level
        /// identified by this method's name with the given `message`.
        pub fn $regular<T: Into<String>>(mut self, message: T) -> Diagnostic {
            self.children.push(Diagnostic::new($level, message));
            self
        }
    };
}

/// Iterator over the children diagnostics of a `Diagnostic`.
#[derive(Debug, Clone)]
pub struct Children<'a>(std::slice::Iter<'a, Diagnostic>);

impl<'a> Iterator for Children<'a> {
    type Item = &'a Diagnostic;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl Diagnostic {
    /// Creates a new diagnostic with the given `level` and `message`.
    pub fn new<T: Into<String>>(level: Level, message: T) -> Diagnostic {
        Diagnostic { level, message: message.into(), spans: vec![], children: vec![] }
    }

    /// Creates a new diagnostic with the given `level` and `message` pointing to
    /// the given set of `spans`.
    pub fn spanned<S, T>(spans: S, level: Level, message: T) -> Diagnostic
    where
        S: MultiSpan,
        T: Into<String>,
    {
        Diagnostic { level, message: message.into(), spans: spans.into_spans(), children: vec![] }
    }

    diagnostic_child_methods!(span_error, error, Level::Error);
    diagnostic_child_methods!(span_warning, warning, Level::Warning);
    diagnostic_child_methods!(span_note, note, Level::Note);
    diagnostic_child_methods!(span_help, help, Level::Help);

    /// Returns the diagnostic `level` for `self`.
    pub fn level(&self) -> Level {
        self.level
    }

    /// Sets the level in `self` to `level`.
    pub fn set_level(&mut self, level: Level) {
        self.level = level;
    }

    /// Returns the message in `self`.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Sets the message in `self` to `message`.
    pub fn set_message<T: Into<String>>(&mut self, message: T) {
        self.message = message.into();
    }

    /// Returns the `Span`s in `self`.
    pub fn spans(&self) -> &[Span] {
        &self.spans
    }

    /// Sets the `Span`s in `self` to `spans`.
    pub fn set_spans<S: MultiSpan>(&mut self, spans: S) {
        self.spans = spans.into_spans();
    }

    /// Returns an iterator over the children diagnostics of `self`.
    pub fn children(&self) -> Children<'_> {
        Children(self.children.iter())
    }

    /// Emit the diagnostic.
    pub fn emit(self) {
        fn to_internal(spans: Vec<Span>) -> super::bridge::client::MultiSpan {
            let mut multi_span = super::bridge::client::MultiSpan::new();
            for span in spans {
                multi_span.push(span.0);
            }
            multi_span
        }

        let mut diag = super::bridge::client::Diagnostic::new(
            self.level,
            &self.message[..],
            to_internal(self.spans),
        );
        for c in self.children {
            diag.sub(c.level, &c.message[..], to_internal(c.spans));
        }
        diag.emit();
    }
}
