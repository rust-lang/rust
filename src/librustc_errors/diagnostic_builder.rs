use crate::Diagnostic;
use crate::DiagnosticId;
use crate::DiagnosticStyledString;
use crate::Applicability;

use crate::Level;
use crate::Handler;
use std::fmt::{self, Debug};
use std::ops::{Deref, DerefMut};
use std::thread::panicking;
use syntax_pos::{MultiSpan, Span};
use log::debug;

/// Used for emitting structured error messages and other diagnostic information.
///
/// If there is some state in a downstream crate you would like to
/// access in the methods of `DiagnosticBuilder` here, consider
/// extending `HandlerFlags`, accessed via `self.handler.flags`.
#[must_use]
#[derive(Clone)]
pub struct DiagnosticBuilder<'a> {
    pub handler: &'a Handler,
    diagnostic: Diagnostic,
    allow_suggestions: bool,
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
        pub fn $n(&mut self, $($name: $ty),*) -> &mut Self {
            self.diagnostic.$n($($name),*);
            self
        }
    };

    // Forward pattern for &mut self -> &mut Self, with S: Into<MultiSpan>
    // type parameter. No obvious way to make this more generic.
    (
        $(#[$attrs:meta])*
        pub fn $n:ident<S: Into<MultiSpan>>(
            &mut self,
            $($name:ident: $ty:ty),*
            $(,)?
        ) -> &mut Self
    ) => {
        $(#[$attrs])*
        pub fn $n<S: Into<MultiSpan>>(&mut self, $($name: $ty),*) -> &mut Self {
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
        if self.cancelled() {
            return;
        }

        self.handler.emit_db(&self);
        self.cancel();
    }

    /// Emit the diagnostic unless `delay` is true,
    /// in which case the emission will be delayed as a bug.
    ///
    /// See `emit` and `delay_as_bug` for details.
    pub fn emit_unless(&mut self, delay: bool) {
        if delay {
            self.delay_as_bug()
        } else {
            self.emit()
        }
    }

    /// Buffers the diagnostic for later emission, unless handler
    /// has disabled such buffering.
    pub fn buffer(mut self, buffered_diagnostics: &mut Vec<Diagnostic>) {
        if self.handler.flags.dont_buffer_diagnostics ||
            self.handler.flags.treat_err_as_bug.is_some()
        {
            self.emit();
            return;
        }

        // We need to use `ptr::read` because `DiagnosticBuilder`
        // implements `Drop`.
        let diagnostic;
        unsafe {
            diagnostic = std::ptr::read(&self.diagnostic);
            std::mem::forget(self);
        };
        // Logging here is useful to help track down where in logs an error was
        // actually emitted.
        debug!("buffer: diagnostic={:?}", diagnostic);
        buffered_diagnostics.push(diagnostic);
    }

    /// Convenience function for internal use, clients should use one of the
    /// span_* methods instead.
    pub fn sub<S: Into<MultiSpan>>(
        &mut self,
        level: Level,
        message: &str,
        span: Option<S>,
    ) -> &mut Self {
        let span = span.map(|s| s.into()).unwrap_or_else(|| MultiSpan::new());
        self.diagnostic.sub(level, message, span, None);
        self
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
        self.handler.delay_as_bug(self.diagnostic.clone());
        self.cancel();
    }

    /// Adds a span/label to be included in the resulting snippet.
    /// This is pushed onto the `MultiSpan` that was created when the
    /// diagnostic was first built. If you don't call this function at
    /// all, and you just supplied a `Span` to create the diagnostic,
    /// then the snippet will just include that `Span`, which is
    /// called the primary span.
    pub fn span_label<T: Into<String>>(&mut self, span: Span, label: T) -> &mut Self {
        self.diagnostic.span_label(span, label);
        self
    }

    forward!(pub fn note_expected_found(&mut self,
                                        label: &dyn fmt::Display,
                                        expected: DiagnosticStyledString,
                                        found: DiagnosticStyledString,
                                        ) -> &mut Self);

    forward!(pub fn note_expected_found_extra(&mut self,
                                              label: &dyn fmt::Display,
                                              expected: DiagnosticStyledString,
                                              found: DiagnosticStyledString,
                                              expected_extra: &dyn fmt::Display,
                                              found_extra: &dyn fmt::Display,
                                              ) -> &mut Self);

    forward!(pub fn note(&mut self, msg: &str) -> &mut Self);
    forward!(pub fn span_note<S: Into<MultiSpan>>(&mut self,
                                                  sp: S,
                                                  msg: &str,
                                                  ) -> &mut Self);
    forward!(pub fn warn(&mut self, msg: &str) -> &mut Self);
    forward!(pub fn span_warn<S: Into<MultiSpan>>(&mut self, sp: S, msg: &str) -> &mut Self);
    forward!(pub fn help(&mut self, msg: &str) -> &mut Self);
    forward!(pub fn span_help<S: Into<MultiSpan>>(&mut self,
                                                  sp: S,
                                                  msg: &str,
                                                  ) -> &mut Self);

    pub fn multipart_suggestion(
        &mut self,
        msg: &str,
        suggestion: Vec<(Span, String)>,
        applicability: Applicability,
    ) -> &mut Self {
        if !self.allow_suggestions {
            return self
        }
        self.diagnostic.multipart_suggestion(
            msg,
            suggestion,
            applicability,
        );
        self
    }

    pub fn tool_only_multipart_suggestion(
        &mut self,
        msg: &str,
        suggestion: Vec<(Span, String)>,
        applicability: Applicability,
    ) -> &mut Self {
        if !self.allow_suggestions {
            return self
        }
        self.diagnostic.tool_only_multipart_suggestion(
            msg,
            suggestion,
            applicability,
        );
        self
    }


    pub fn span_suggestion(
        &mut self,
        sp: Span,
        msg: &str,
        suggestion: String,
        applicability: Applicability,
    ) -> &mut Self {
        if !self.allow_suggestions {
            return self
        }
        self.diagnostic.span_suggestion(
            sp,
            msg,
            suggestion,
            applicability,
        );
        self
    }

    pub fn span_suggestions(
        &mut self,
        sp: Span,
        msg: &str,
        suggestions: impl Iterator<Item = String>,
        applicability: Applicability,
    ) -> &mut Self {
        if !self.allow_suggestions {
            return self
        }
        self.diagnostic.span_suggestions(
            sp,
            msg,
            suggestions,
            applicability,
        );
        self
    }

    pub fn span_suggestion_short(
        &mut self,
        sp: Span,
        msg: &str,
        suggestion: String,
        applicability: Applicability,
    ) -> &mut Self {
        if !self.allow_suggestions {
            return self
        }
        self.diagnostic.span_suggestion_short(
            sp,
            msg,
            suggestion,
            applicability,
        );
        self
    }

    pub fn span_suggestion_hidden(
        &mut self,
        sp: Span,
        msg: &str,
        suggestion: String,
        applicability: Applicability,
    ) -> &mut Self {
        if !self.allow_suggestions {
            return self
        }
        self.diagnostic.span_suggestion_hidden(
            sp,
            msg,
            suggestion,
            applicability,
        );
        self
    }

    pub fn tool_only_span_suggestion(
        &mut self,
        sp: Span,
        msg: &str,
        suggestion: String,
        applicability: Applicability,
    ) -> &mut Self {
        if !self.allow_suggestions {
            return self
        }
        self.diagnostic.tool_only_span_suggestion(
            sp,
            msg,
            suggestion,
            applicability,
        );
        self
    }

    forward!(pub fn set_span<S: Into<MultiSpan>>(&mut self, sp: S) -> &mut Self);
    forward!(pub fn code(&mut self, s: DiagnosticId) -> &mut Self);

    pub fn allow_suggestions(&mut self, allow: bool) -> &mut Self {
        self.allow_suggestions = allow;
        self
    }

    /// Convenience function for internal use, clients should use one of the
    /// struct_* methods on Handler.
    pub fn new(handler: &'a Handler, level: Level, message: &str) -> DiagnosticBuilder<'a> {
        DiagnosticBuilder::new_with_code(handler, level, None, message)
    }

    /// Convenience function for internal use, clients should use one of the
    /// struct_* methods on Handler.
    crate fn new_with_code(handler: &'a Handler,
                         level: Level,
                         code: Option<DiagnosticId>,
                         message: &str)
                         -> DiagnosticBuilder<'a> {
        let diagnostic = Diagnostic::new_with_code(level, code, message);
        DiagnosticBuilder::new_diagnostic(handler, diagnostic)
    }

    /// Creates a new `DiagnosticBuilder` with an already constructed
    /// diagnostic.
    pub fn new_diagnostic(handler: &'a Handler, diagnostic: Diagnostic)
                         -> DiagnosticBuilder<'a> {
        DiagnosticBuilder {
            handler,
            diagnostic,
            allow_suggestions: true,
        }
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
            let mut db = DiagnosticBuilder::new(self.handler,
                                                Level::Bug,
                                                "Error constructed but not emitted");
            db.emit();
            panic!();
        }
    }
}
