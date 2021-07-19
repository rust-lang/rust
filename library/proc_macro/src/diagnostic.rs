use crate::Span;
use std::iter;

/// An enum representing a diagnostic level.
#[doc(hidden)]
#[unstable(feature = "proc_macro_internals", issue = "27812")]
#[derive(Copy, Clone, Debug)]
#[non_exhaustive]
pub enum Level {
    /// An error.
    Error,
    /// A note.
    Note,
    /// A help message.
    Help,
}

/// Trait implemented by types that can be converted into an iterator of [`Span`]s.
#[unstable(feature = "proc_macro_diagnostic", issue = "54140")]
pub trait Spanned {
    /// The concrete iterator type returned by `spans`.
    type Iter: Iterator<Item = Span>;

    /// Converts `self` into an iterator of `Span`.
    fn spans(self) -> Self::Iter;

    /// Best-effort join of all `Span`s in the iterator.
    fn span(self) -> Span
    where
        Self: Sized,
    {
        self.spans()
            .reduce(|a, b| match a.join(b) {
                Some(joined) => joined,
                // Skip any bad joins.
                None => a,
            })
            .unwrap_or_else(Span::call_site)
    }

    /// Create an error attached to the `Span`.
    fn error(self, msg: &str) -> Diagnostic
    where
        Self: Sized,
    {
        Diagnostic::error(msg).mark_all(self.spans())
    }

    /// Create an note attached to the `Span`.
    fn note(self, msg: &str) -> Diagnostic
    where
        Self: Sized,
    {
        Diagnostic::note(msg).mark_all(self.spans())
    }
}

#[unstable(feature = "proc_macro_diagnostic", issue = "54140")]
impl<S: Spanned, I: IntoIterator<Item = S>> Spanned for I {
    type Iter = impl Iterator<Item = Span>;

    fn spans(self) -> Self::Iter {
        self.into_iter().flat_map(Spanned::spans)
    }
}

#[unstable(feature = "proc_macro_diagnostic", issue = "54140")]
impl Spanned for Span {
    type Iter = iter::Once<Self>;

    fn spans(self) -> Self::Iter {
        iter::once(self)
    }

    fn span(self) -> Span {
        self
    }
}

macro_rules! impl_span_passthrough {
    ($($type:ty)*) => {$(
        #[unstable(feature = "proc_macro_diagnostic", issue = "54140")]
        impl Spanned for $type {
            type Iter = iter::Once<Span>;

            fn spans(self) -> Self::Iter {
                iter::once(self.span())
            }

            fn span(self) -> Span {
                Self::span(&self)
            }
        }
    )*}
}

impl_span_passthrough![
    crate::Group
    crate::Ident
    crate::Literal
    crate::Punct
    crate::TokenTree
];

/// A structure representing a diagnostic message and associated children
/// messages.
#[unstable(feature = "proc_macro_diagnostic", issue = "54140")]
#[must_use = "diagnostics do nothing unless emitted"]
#[derive(Clone, Debug)]
pub struct Diagnostic {
    level: Level,
    message: String,
    spans: Vec<Span>,
    labels: Vec<(Span, String)>,
    children: Vec<Diagnostic>,
}

#[unstable(feature = "proc_macro_diagnostic", issue = "54140")]
impl Diagnostic {
    /// Obtain a mutable reference to the last message.
    fn last_message_mut(&mut self) -> &mut Diagnostic {
        if self.children.is_empty() { self } else { self.children.last_mut().unwrap() }
    }

    /// Creates a new error diagnostic with the provided message.
    #[unstable(feature = "proc_macro_diagnostic", issue = "54140")]
    pub fn error(message: &str) -> Self {
        Diagnostic {
            level: Level::Error,
            message: message.into(),
            spans: vec![],
            labels: vec![],
            children: vec![],
        }
    }

    // FIXME Add lint-associated warnings

    /// Creates a new note diagnostic with the provided message.
    #[unstable(feature = "proc_macro_diagnostic", issue = "54140")]
    pub fn note(message: &str) -> Self {
        Diagnostic {
            level: Level::Note,
            message: message.into(),
            spans: vec![],
            labels: vec![],
            children: vec![],
        }
    }

    /// Adds a help message to the diagnostic.
    #[unstable(feature = "proc_macro_diagnostic", issue = "54140")]
    pub fn with_help(mut self, message: &str) -> Self {
        self.children.push(Self {
            level: Level::Help,
            message: message.into(),
            spans: vec![],
            labels: vec![],
            children: vec![],
        });
        self
    }

    /// Adds a note message to the diagnostic.
    #[unstable(feature = "proc_macro_diagnostic", issue = "54140")]
    pub fn with_note(mut self, message: &str) -> Self {
        self.children.push(Self {
            level: Level::Note,
            message: message.into(),
            spans: vec![],
            labels: vec![],
            children: vec![],
        });
        self
    }

    /// Adds one mark with the given `item.span()` to the last message. The mark
    /// is "primary" if no other mark or label has been applied to the previous
    /// message and "secondary" otherwise.
    #[unstable(feature = "proc_macro_diagnostic", issue = "54140")]
    pub fn mark<S: Spanned>(mut self, item: S) -> Self {
        self.last_message_mut().spans.push(item.span());
        self
    }

    /// Adds a spanned mark for every span in `item.spans()` to the last
    /// message. The marks are "primary" if no other mark or label has been
    /// applied to the previous message and "secondary" otherwise.
    #[unstable(feature = "proc_macro_diagnostic", issue = "54140")]
    pub fn mark_all<S: Spanned>(mut self, item: S) -> Self {
        self.last_message_mut().spans.extend(item.spans());
        self
    }

    /// Adds one spanned mark with a label `msg` with the given `item.span()` to
    /// the last message. The mark is "primary" if no other mark or label has
    /// been applied to the previous message and "secondary" otherwise.
    #[unstable(feature = "proc_macro_diagnostic", issue = "54140")]
    pub fn label<S: Spanned>(mut self, item: S, message: &str) -> Self {
        self.last_message_mut().labels.push((item.span(), message.into()));
        self
    }

    /// Emit the diagnostic.
    #[unstable(feature = "proc_macro_diagnostic", issue = "54140")]
    pub fn emit(self) {
        fn to_internal(
            spans: Vec<Span>,
            labels: Vec<(Span, String)>,
        ) -> crate::bridge::client::MultiSpan {
            let mut multi_span = crate::bridge::client::MultiSpan::new();
            for span in spans {
                multi_span.push_primary_span(span.0);
            }
            for (span, label) in labels {
                multi_span.push_span_label(span.0, label);
            }
            multi_span
        }

        let mut diag = crate::bridge::client::Diagnostic::new(
            self.level,
            &self.message[..],
            to_internal(self.spans, self.labels),
        );
        for c in self.children {
            diag.sub(c.level, &c.message[..], to_internal(c.spans, c.labels));
        }
        diag.emit();
    }
}
