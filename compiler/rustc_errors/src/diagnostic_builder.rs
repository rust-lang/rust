use crate::{Diagnostic, DiagnosticId, DiagnosticStyledString};
use crate::{Handler, Level, StashKey};
use rustc_lint_defs::Applicability;

use rustc_span::{MultiSpan, Span};
use std::fmt::{self, Debug};
use std::ops::{Deref, DerefMut};
use std::thread::panicking;
use tracing::debug;

/// Used for emitting structured error messages and other diagnostic information.
///
/// If there is some state in a downstream crate you would like to
/// access in the methods of `DiagnosticBuilder` here, consider
/// extending `HandlerFlags`, accessed via `self.handler.flags`.
#[must_use]
#[derive(Clone)]
pub struct DiagnosticBuilder<'a> {
    handler: &'a Handler,

    /// `Diagnostic` is a large type, and `DiagnosticBuilder` is often used as a
    /// return value, especially within the frequently-used `PResult` type.
    /// In theory, return value optimization (RVO) should avoid unnecessary
    /// copying. In practice, it does not (at the time of writing).
    diagnostic: Box<Diagnostic>,
}

/// In general, the `DiagnosticBuilder` uses deref to allow access to
/// the fields and methods of the embedded `diagnostic` in a
/// transparent way. *However,* many of the methods are intended to
/// be used in a chained way, and hence ought to return `self`. In
/// that case, we can't just naively forward to the method on the
/// `diagnostic`, because the return type would be a `&Diagnostic`
/// instead of a `&DiagnosticBuilder<'a>`. This `forward!` macro makes
/// it easy to declare such methods on the builder.
macro_rules! forward {
    // Forward pattern for &self -> &Self
    (
        $(#[$attrs:meta])*
        pub fn $n:ident(&self, $($name:ident: $ty:ty),* $(,)?) -> &Self
    ) => {
        $(#[$attrs])*
        #[doc = concat!("See [`Diagnostic::", stringify!($n), "()`].")]
        pub fn $n(&self, $($name: $ty),*) -> &Self {
            self.diagnostic.$n($($name),*);
            self
        }
    };

    // Forward pattern for &mut self -> &mut Self
    (
        $(#[$attrs:meta])*
        pub fn $n:ident(&mut self, $($name:ident: $ty:ty),* $(,)?) -> &mut Self
    ) => {
        $(#[$attrs])*
        #[doc = concat!("See [`Diagnostic::", stringify!($n), "()`].")]
        pub fn $n(&mut self, $($name: $ty),*) -> &mut Self {
            self.diagnostic.$n($($name),*);
            self
        }
    };

    // Forward pattern for &mut self -> &mut Self, with generic parameters.
    (
        $(#[$attrs:meta])*
        pub fn $n:ident<$($generic:ident: $bound:path),*>(
            &mut self,
            $($name:ident: $ty:ty),*
            $(,)?
        ) -> &mut Self
    ) => {
        $(#[$attrs])*
        #[doc = concat!("See [`Diagnostic::", stringify!($n), "()`].")]
        pub fn $n<$($generic: $bound),*>(&mut self, $($name: $ty),*) -> &mut Self {
            self.diagnostic.$n($($name),*);
            self
        }
    };
}

impl<'a> Deref for DiagnosticBuilder<'a> {
    type Target = Diagnostic;

    fn deref(&self) -> &Diagnostic {
        &self.diagnostic
    }
}

impl<'a> DerefMut for DiagnosticBuilder<'a> {
    fn deref_mut(&mut self) -> &mut Diagnostic {
        &mut self.diagnostic
    }
}

impl<'a> DiagnosticBuilder<'a> {
    /// Emit the diagnostic.
    pub fn emit(&mut self) {
        self.handler.emit_diagnostic(&self);
        self.cancel();
    }

    /// Emit the diagnostic unless `delay` is true,
    /// in which case the emission will be delayed as a bug.
    ///
    /// See `emit` and `delay_as_bug` for details.
    pub fn emit_unless(&mut self, delay: bool) {
        if delay {
            self.delay_as_bug();
        } else {
            self.emit();
        }
    }

    /// Stashes diagnostic for possible later improvement in a different,
    /// later stage of the compiler. The diagnostic can be accessed with
    /// the provided `span` and `key` through [`Handler::steal_diagnostic()`].
    ///
    /// As with `buffer`, this is unless the handler has disabled such buffering.
    pub fn stash(self, span: Span, key: StashKey) {
        if let Some((diag, handler)) = self.into_diagnostic() {
            handler.stash_diagnostic(span, key, diag);
        }
    }

    /// Converts the builder to a `Diagnostic` for later emission,
    /// unless handler has disabled such buffering.
    pub fn into_diagnostic(mut self) -> Option<(Diagnostic, &'a Handler)> {
        if self.handler.flags.dont_buffer_diagnostics
            || self.handler.flags.treat_err_as_bug.is_some()
        {
            self.emit();
            return None;
        }

        let handler = self.handler;

        // We must use `Level::Cancelled` for `dummy` to avoid an ICE about an
        // unused diagnostic.
        let dummy = Diagnostic::new(Level::Cancelled, "");
        let diagnostic = std::mem::replace(&mut *self.diagnostic, dummy);

        // Logging here is useful to help track down where in logs an error was
        // actually emitted.
        debug!("buffer: diagnostic={:?}", diagnostic);

        Some((diagnostic, handler))
    }

    /// Buffers the diagnostic for later emission,
    /// unless handler has disabled such buffering.
    pub fn buffer(self, buffered_diagnostics: &mut Vec<Diagnostic>) {
        buffered_diagnostics.extend(self.into_diagnostic().map(|(diag, _)| diag));
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
    pub fn delay_as_bug(&mut self) {
        self.level = Level::Bug;
        self.handler.delay_as_bug((*self.diagnostic).clone());
        self.cancel();
    }

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
    pub fn span_label(&mut self, span: Span, label: impl Into<String>) -> &mut Self {
        self.diagnostic.span_label(span, label);
        self
    }

    /// Labels all the given spans with the provided label.
    /// See [`Diagnostic::span_label()`] for more information.
    pub fn span_labels(
        &mut self,
        spans: impl IntoIterator<Item = Span>,
        label: impl AsRef<str>,
    ) -> &mut Self {
        let label = label.as_ref();
        for span in spans {
            self.diagnostic.span_label(span, label);
        }
        self
    }

    forward!(pub fn note_expected_found(
        &mut self,
        expected_label: &dyn fmt::Display,
        expected: DiagnosticStyledString,
        found_label: &dyn fmt::Display,
        found: DiagnosticStyledString,
    ) -> &mut Self);

    forward!(pub fn note_expected_found_extra(
        &mut self,
        expected_label: &dyn fmt::Display,
        expected: DiagnosticStyledString,
        found_label: &dyn fmt::Display,
        found: DiagnosticStyledString,
        expected_extra: &dyn fmt::Display,
        found_extra: &dyn fmt::Display,
    ) -> &mut Self);

    forward!(pub fn note_unsuccessful_coercion(
        &mut self,
        expected: DiagnosticStyledString,
        found: DiagnosticStyledString,
    ) -> &mut Self);

    forward!(pub fn note(&mut self, msg: &str) -> &mut Self);
    forward!(pub fn span_note<S: Into<MultiSpan>>(
        &mut self,
        sp: S,
        msg: &str,
    ) -> &mut Self);
    forward!(pub fn warn(&mut self, msg: &str) -> &mut Self);
    forward!(pub fn span_warn<S: Into<MultiSpan>>(&mut self, sp: S, msg: &str) -> &mut Self);
    forward!(pub fn help(&mut self, msg: &str) -> &mut Self);
    forward!(pub fn span_help<S: Into<MultiSpan>>(
        &mut self,
        sp: S,
        msg: &str,
    ) -> &mut Self);
    forward!(pub fn set_is_lint(&mut self,) -> &mut Self);

    forward!(pub fn disable_suggestions(&mut self,) -> &mut Self);

    forward!(pub fn multipart_suggestion(
        &mut self,
        msg: &str,
        suggestion: Vec<(Span, String)>,
        applicability: Applicability,
    ) -> &mut Self);
    forward!(pub fn multipart_suggestion_verbose(
        &mut self,
        msg: &str,
        suggestion: Vec<(Span, String)>,
        applicability: Applicability,
    ) -> &mut Self);
    forward!(pub fn tool_only_multipart_suggestion(
        &mut self,
        msg: &str,
        suggestion: Vec<(Span, String)>,
        applicability: Applicability,
    ) -> &mut Self);
    forward!(pub fn span_suggestion(
        &mut self,
        sp: Span,
        msg: &str,
        suggestion: String,
        applicability: Applicability,
    ) -> &mut Self);
    forward!(pub fn span_suggestions(
        &mut self,
        sp: Span,
        msg: &str,
        suggestions: impl Iterator<Item = String>,
        applicability: Applicability,
    ) -> &mut Self);
    forward!(pub fn multipart_suggestions(
        &mut self,
        msg: &str,
        suggestions: impl Iterator<Item = Vec<(Span, String)>>,
        applicability: Applicability,
    ) -> &mut Self);
    forward!(pub fn span_suggestion_short(
        &mut self,
        sp: Span,
        msg: &str,
        suggestion: String,
        applicability: Applicability,
    ) -> &mut Self);
    forward!(pub fn span_suggestion_verbose(
        &mut self,
        sp: Span,
        msg: &str,
        suggestion: String,
        applicability: Applicability,
    ) -> &mut Self);
    forward!(pub fn span_suggestion_hidden(
        &mut self,
        sp: Span,
        msg: &str,
        suggestion: String,
        applicability: Applicability,
    ) -> &mut Self);
    forward!(pub fn tool_only_span_suggestion(
        &mut self,
        sp: Span,
        msg: &str,
        suggestion: String,
        applicability: Applicability,
    ) -> &mut Self);

    forward!(pub fn set_primary_message<M: Into<String>>(&mut self, msg: M) -> &mut Self);
    forward!(pub fn set_span<S: Into<MultiSpan>>(&mut self, sp: S) -> &mut Self);
    forward!(pub fn code(&mut self, s: DiagnosticId) -> &mut Self);

    /// Convenience function for internal use, clients should use one of the
    /// `struct_*` methods on [`Handler`].
    crate fn new(handler: &'a Handler, level: Level, message: &str) -> DiagnosticBuilder<'a> {
        DiagnosticBuilder::new_with_code(handler, level, None, message)
    }

    /// Convenience function for internal use, clients should use one of the
    /// `struct_*` methods on [`Handler`].
    crate fn new_with_code(
        handler: &'a Handler,
        level: Level,
        code: Option<DiagnosticId>,
        message: &str,
    ) -> DiagnosticBuilder<'a> {
        let diagnostic = Diagnostic::new_with_code(level, code, message);
        DiagnosticBuilder::new_diagnostic(handler, diagnostic)
    }

    /// Creates a new `DiagnosticBuilder` with an already constructed
    /// diagnostic.
    crate fn new_diagnostic(handler: &'a Handler, diagnostic: Diagnostic) -> DiagnosticBuilder<'a> {
        debug!("Created new diagnostic");
        DiagnosticBuilder { handler, diagnostic: Box::new(diagnostic) }
    }
}

impl<'a> Debug for DiagnosticBuilder<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.diagnostic.fmt(f)
    }
}

/// Destructor bomb - a `DiagnosticBuilder` must be either emitted or canceled
/// or we emit a bug.
impl<'a> Drop for DiagnosticBuilder<'a> {
    fn drop(&mut self) {
        if !panicking() && !self.cancelled() {
            let mut db = DiagnosticBuilder::new(
                self.handler,
                Level::Bug,
                "the following error was constructed but not emitted",
            );
            db.emit();
            self.emit();
            panic!();
        }
    }
}

#[macro_export]
macro_rules! struct_span_err {
    ($session:expr, $span:expr, $code:ident, $($message:tt)*) => ({
        $session.struct_span_err_with_code(
            $span,
            &format!($($message)*),
            $crate::error_code!($code),
        )
    })
}

#[macro_export]
macro_rules! error_code {
    ($code:ident) => {{ $crate::DiagnosticId::Error(stringify!($code).to_owned()) }};
}
