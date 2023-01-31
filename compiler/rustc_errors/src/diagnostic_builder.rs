use crate::diagnostic::IntoDiagnosticArg;
use crate::{
    Diagnostic, DiagnosticId, DiagnosticMessage, DiagnosticStyledString, ErrorGuaranteed,
    ExplicitBug, SubdiagnosticMessage,
};
use crate::{Handler, Level, MultiSpan, StashKey};
use rustc_lint_defs::Applicability;
use rustc_span::source_map::Spanned;

use rustc_span::Span;
use std::borrow::Cow;
use std::fmt::{self, Debug};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::panic;
use std::thread::panicking;

/// Trait implemented by error types. This should not be implemented manually. Instead, use
/// `#[derive(Diagnostic)]` -- see [rustc_macros::Diagnostic].
#[rustc_diagnostic_item = "IntoDiagnostic"]
pub trait IntoDiagnostic<'a, T: EmissionGuarantee = ErrorGuaranteed> {
    /// Write out as a diagnostic out of `Handler`.
    #[must_use]
    fn into_diagnostic(self, handler: &'a Handler) -> DiagnosticBuilder<'a, T>;
}

impl<'a, T, E> IntoDiagnostic<'a, E> for Spanned<T>
where
    T: IntoDiagnostic<'a, E>,
    E: EmissionGuarantee,
{
    fn into_diagnostic(self, handler: &'a Handler) -> DiagnosticBuilder<'a, E> {
        let mut diag = self.node.into_diagnostic(handler);
        diag.set_span(self.span);
        diag
    }
}

/// Used for emitting structured error messages and other diagnostic information.
///
/// If there is some state in a downstream crate you would like to
/// access in the methods of `DiagnosticBuilder` here, consider
/// extending `HandlerFlags`, accessed via `self.handler.flags`.
#[must_use]
#[derive(Clone)]
pub struct DiagnosticBuilder<'a, G: EmissionGuarantee> {
    inner: DiagnosticBuilderInner<'a>,
    _marker: PhantomData<G>,
}

/// This type exists only for `DiagnosticBuilder::forget_guarantee`, because it:
/// 1. lacks the `G` parameter and therefore `DiagnosticBuilder<G1>` can be
///    converted into `DiagnosticBuilder<G2>` while reusing the `inner` field
/// 2. can implement the `Drop` "bomb" instead of `DiagnosticBuilder`, as it
///    contains all of the data (`state` + `diagnostic`) of `DiagnosticBuilder`
///
/// The `diagnostic` field is not `Copy` and can't be moved out of whichever
/// type implements the `Drop` "bomb", but because of the above two facts, that
/// never needs to happen - instead, the whole `inner: DiagnosticBuilderInner`
/// can be moved out of a `DiagnosticBuilder` and into another.
#[must_use]
#[derive(Clone)]
struct DiagnosticBuilderInner<'a> {
    state: DiagnosticBuilderState<'a>,

    /// `Diagnostic` is a large type, and `DiagnosticBuilder` is often used as a
    /// return value, especially within the frequently-used `PResult` type.
    /// In theory, return value optimization (RVO) should avoid unnecessary
    /// copying. In practice, it does not (at the time of writing).
    diagnostic: Box<Diagnostic>,
}

#[derive(Clone)]
enum DiagnosticBuilderState<'a> {
    /// Initial state of a `DiagnosticBuilder`, before `.emit()` or `.cancel()`.
    ///
    /// The `Diagnostic` will be emitted through this `Handler`.
    Emittable(&'a Handler),

    /// State of a `DiagnosticBuilder`, after `.emit()` or *during* `.cancel()`.
    ///
    /// The `Diagnostic` will be ignored when calling `.emit()`, and it can be
    /// assumed that `.emit()` was previously called, to end up in this state.
    ///
    /// While this is also used by `.cancel()`, this state is only observed by
    /// the `Drop` `impl` of `DiagnosticBuilderInner`, as `.cancel()` takes
    /// `self` by-value specifically to prevent any attempts to `.emit()`.
    ///
    // FIXME(eddyb) currently this doesn't prevent extending the `Diagnostic`,
    // despite that being potentially lossy, if important information is added
    // *after* the original `.emit()` call.
    AlreadyEmittedOrDuringCancellation,
}

// `DiagnosticBuilderState` should be pointer-sized.
rustc_data_structures::static_assert_size!(
    DiagnosticBuilderState<'_>,
    std::mem::size_of::<&Handler>()
);

/// Trait for types that `DiagnosticBuilder::emit` can return as a "guarantee"
/// (or "proof") token that the emission happened.
pub trait EmissionGuarantee: Sized {
    /// Implementation of `DiagnosticBuilder::emit`, fully controlled by each
    /// `impl` of `EmissionGuarantee`, to make it impossible to create a value
    /// of `Self` without actually performing the emission.
    #[track_caller]
    fn diagnostic_builder_emit_producing_guarantee(db: &mut DiagnosticBuilder<'_, Self>) -> Self;

    /// Creates a new `DiagnosticBuilder` that will return this type of guarantee.
    #[track_caller]
    fn make_diagnostic_builder(
        handler: &Handler,
        msg: impl Into<DiagnosticMessage>,
    ) -> DiagnosticBuilder<'_, Self>;
}

/// Private module for sealing the `IsError` helper trait.
mod sealed_level_is_error {
    use crate::Level;

    /// Sealed helper trait for statically checking that a `Level` is an error.
    pub(crate) trait IsError<const L: Level> {}

    impl IsError<{ Level::Bug }> for () {}
    impl IsError<{ Level::DelayedBug }> for () {}
    impl IsError<{ Level::Fatal }> for () {}
    // NOTE(eddyb) `Level::Error { lint: true }` is also an error, but lints
    // don't need error guarantees, as their levels are always dynamic.
    impl IsError<{ Level::Error { lint: false } }> for () {}
}

impl<'a> DiagnosticBuilder<'a, ErrorGuaranteed> {
    /// Convenience function for internal use, clients should use one of the
    /// `struct_*` methods on [`Handler`].
    #[track_caller]
    pub(crate) fn new_guaranteeing_error<M: Into<DiagnosticMessage>, const L: Level>(
        handler: &'a Handler,
        message: M,
    ) -> Self
    where
        (): sealed_level_is_error::IsError<L>,
    {
        Self {
            inner: DiagnosticBuilderInner {
                state: DiagnosticBuilderState::Emittable(handler),
                diagnostic: Box::new(Diagnostic::new_with_code(L, None, message)),
            },
            _marker: PhantomData,
        }
    }

    /// Discard the guarantee `.emit()` would return, in favor of having the
    /// type `DiagnosticBuilder<'a, ()>`. This may be necessary whenever there
    /// is a common codepath handling both errors and warnings.
    pub fn forget_guarantee(self) -> DiagnosticBuilder<'a, ()> {
        DiagnosticBuilder { inner: self.inner, _marker: PhantomData }
    }
}

// FIXME(eddyb) make `ErrorGuaranteed` impossible to create outside `.emit()`.
impl EmissionGuarantee for ErrorGuaranteed {
    fn diagnostic_builder_emit_producing_guarantee(db: &mut DiagnosticBuilder<'_, Self>) -> Self {
        match db.inner.state {
            // First `.emit()` call, the `&Handler` is still available.
            DiagnosticBuilderState::Emittable(handler) => {
                db.inner.state = DiagnosticBuilderState::AlreadyEmittedOrDuringCancellation;

                let guar = handler.emit_diagnostic(&mut db.inner.diagnostic);

                // Only allow a guarantee if the `level` wasn't switched to a
                // non-error - the field isn't `pub`, but the whole `Diagnostic`
                // can be overwritten with a new one, thanks to `DerefMut`.
                assert!(
                    db.inner.diagnostic.is_error(),
                    "emitted non-error ({:?}) diagnostic \
                     from `DiagnosticBuilder<ErrorGuaranteed>`",
                    db.inner.diagnostic.level,
                );
                guar.unwrap()
            }
            // `.emit()` was previously called, disallowed from repeating it,
            // but can take advantage of the previous `.emit()`'s guarantee
            // still being applicable (i.e. as a form of idempotency).
            DiagnosticBuilderState::AlreadyEmittedOrDuringCancellation => {
                // Only allow a guarantee if the `level` wasn't switched to a
                // non-error - the field isn't `pub`, but the whole `Diagnostic`
                // can be overwritten with a new one, thanks to `DerefMut`.
                assert!(
                    db.inner.diagnostic.is_error(),
                    "`DiagnosticBuilder<ErrorGuaranteed>`'s diagnostic \
                     became non-error ({:?}), after original `.emit()`",
                    db.inner.diagnostic.level,
                );
                ErrorGuaranteed::unchecked_claim_error_was_emitted()
            }
        }
    }

    #[track_caller]
    fn make_diagnostic_builder(
        handler: &Handler,
        msg: impl Into<DiagnosticMessage>,
    ) -> DiagnosticBuilder<'_, Self> {
        DiagnosticBuilder::new_guaranteeing_error::<_, { Level::Error { lint: false } }>(
            handler, msg,
        )
    }
}

impl<'a> DiagnosticBuilder<'a, ()> {
    /// Convenience function for internal use, clients should use one of the
    /// `struct_*` methods on [`Handler`].
    #[track_caller]
    pub(crate) fn new<M: Into<DiagnosticMessage>>(
        handler: &'a Handler,
        level: Level,
        message: M,
    ) -> Self {
        let diagnostic = Diagnostic::new_with_code(level, None, message);
        Self::new_diagnostic(handler, diagnostic)
    }

    /// Creates a new `DiagnosticBuilder` with an already constructed
    /// diagnostic.
    #[track_caller]
    pub(crate) fn new_diagnostic(handler: &'a Handler, diagnostic: Diagnostic) -> Self {
        debug!("Created new diagnostic");
        Self {
            inner: DiagnosticBuilderInner {
                state: DiagnosticBuilderState::Emittable(handler),
                diagnostic: Box::new(diagnostic),
            },
            _marker: PhantomData,
        }
    }
}

// FIXME(eddyb) should there be a `Option<ErrorGuaranteed>` impl as well?
impl EmissionGuarantee for () {
    fn diagnostic_builder_emit_producing_guarantee(db: &mut DiagnosticBuilder<'_, Self>) -> Self {
        match db.inner.state {
            // First `.emit()` call, the `&Handler` is still available.
            DiagnosticBuilderState::Emittable(handler) => {
                db.inner.state = DiagnosticBuilderState::AlreadyEmittedOrDuringCancellation;

                handler.emit_diagnostic(&mut db.inner.diagnostic);
            }
            // `.emit()` was previously called, disallowed from repeating it.
            DiagnosticBuilderState::AlreadyEmittedOrDuringCancellation => {}
        }
    }

    fn make_diagnostic_builder(
        handler: &Handler,
        msg: impl Into<DiagnosticMessage>,
    ) -> DiagnosticBuilder<'_, Self> {
        DiagnosticBuilder::new(handler, Level::Warning(None), msg)
    }
}

/// Marker type which enables implementation of `create_note` and `emit_note` functions for
/// note-without-error struct diagnostics.
#[derive(Copy, Clone)]
pub struct Noted;

impl<'a> DiagnosticBuilder<'a, Noted> {
    /// Convenience function for internal use, clients should use one of the
    /// `struct_*` methods on [`Handler`].
    pub(crate) fn new_note(handler: &'a Handler, message: impl Into<DiagnosticMessage>) -> Self {
        let diagnostic = Diagnostic::new_with_code(Level::Note, None, message);
        Self::new_diagnostic_note(handler, diagnostic)
    }

    /// Creates a new `DiagnosticBuilder` with an already constructed
    /// diagnostic.
    pub(crate) fn new_diagnostic_note(handler: &'a Handler, diagnostic: Diagnostic) -> Self {
        debug!("Created new diagnostic");
        Self {
            inner: DiagnosticBuilderInner {
                state: DiagnosticBuilderState::Emittable(handler),
                diagnostic: Box::new(diagnostic),
            },
            _marker: PhantomData,
        }
    }
}

impl EmissionGuarantee for Noted {
    fn diagnostic_builder_emit_producing_guarantee(db: &mut DiagnosticBuilder<'_, Self>) -> Self {
        match db.inner.state {
            // First `.emit()` call, the `&Handler` is still available.
            DiagnosticBuilderState::Emittable(handler) => {
                db.inner.state = DiagnosticBuilderState::AlreadyEmittedOrDuringCancellation;
                handler.emit_diagnostic(&mut db.inner.diagnostic);
            }
            // `.emit()` was previously called, disallowed from repeating it.
            DiagnosticBuilderState::AlreadyEmittedOrDuringCancellation => {}
        }

        Noted
    }

    fn make_diagnostic_builder(
        handler: &Handler,
        msg: impl Into<DiagnosticMessage>,
    ) -> DiagnosticBuilder<'_, Self> {
        DiagnosticBuilder::new_note(handler, msg)
    }
}

/// Marker type which enables implementation of `create_bug` and `emit_bug` functions for
/// bug struct diagnostics.
#[derive(Copy, Clone)]
pub struct Bug;

impl<'a> DiagnosticBuilder<'a, Bug> {
    /// Convenience function for internal use, clients should use one of the
    /// `struct_*` methods on [`Handler`].
    #[track_caller]
    pub(crate) fn new_bug(handler: &'a Handler, message: impl Into<DiagnosticMessage>) -> Self {
        let diagnostic = Diagnostic::new_with_code(Level::Bug, None, message);
        Self::new_diagnostic_bug(handler, diagnostic)
    }

    /// Creates a new `DiagnosticBuilder` with an already constructed
    /// diagnostic.
    pub(crate) fn new_diagnostic_bug(handler: &'a Handler, diagnostic: Diagnostic) -> Self {
        debug!("Created new diagnostic bug");
        Self {
            inner: DiagnosticBuilderInner {
                state: DiagnosticBuilderState::Emittable(handler),
                diagnostic: Box::new(diagnostic),
            },
            _marker: PhantomData,
        }
    }
}

impl EmissionGuarantee for Bug {
    fn diagnostic_builder_emit_producing_guarantee(db: &mut DiagnosticBuilder<'_, Self>) -> Self {
        match db.inner.state {
            // First `.emit()` call, the `&Handler` is still available.
            DiagnosticBuilderState::Emittable(handler) => {
                db.inner.state = DiagnosticBuilderState::AlreadyEmittedOrDuringCancellation;

                handler.emit_diagnostic(&mut db.inner.diagnostic);
            }
            // `.emit()` was previously called, disallowed from repeating it.
            DiagnosticBuilderState::AlreadyEmittedOrDuringCancellation => {}
        }
        // Then panic. No need to return the marker type.
        panic::panic_any(ExplicitBug);
    }

    fn make_diagnostic_builder(
        handler: &Handler,
        msg: impl Into<DiagnosticMessage>,
    ) -> DiagnosticBuilder<'_, Self> {
        DiagnosticBuilder::new_bug(handler, msg)
    }
}

impl<'a> DiagnosticBuilder<'a, !> {
    /// Convenience function for internal use, clients should use one of the
    /// `struct_*` methods on [`Handler`].
    #[track_caller]
    pub(crate) fn new_fatal(handler: &'a Handler, message: impl Into<DiagnosticMessage>) -> Self {
        let diagnostic = Diagnostic::new_with_code(Level::Fatal, None, message);
        Self::new_diagnostic_fatal(handler, diagnostic)
    }

    /// Creates a new `DiagnosticBuilder` with an already constructed
    /// diagnostic.
    pub(crate) fn new_diagnostic_fatal(handler: &'a Handler, diagnostic: Diagnostic) -> Self {
        debug!("Created new diagnostic");
        Self {
            inner: DiagnosticBuilderInner {
                state: DiagnosticBuilderState::Emittable(handler),
                diagnostic: Box::new(diagnostic),
            },
            _marker: PhantomData,
        }
    }
}

impl EmissionGuarantee for ! {
    fn diagnostic_builder_emit_producing_guarantee(db: &mut DiagnosticBuilder<'_, Self>) -> Self {
        match db.inner.state {
            // First `.emit()` call, the `&Handler` is still available.
            DiagnosticBuilderState::Emittable(handler) => {
                db.inner.state = DiagnosticBuilderState::AlreadyEmittedOrDuringCancellation;

                handler.emit_diagnostic(&mut db.inner.diagnostic);
            }
            // `.emit()` was previously called, disallowed from repeating it.
            DiagnosticBuilderState::AlreadyEmittedOrDuringCancellation => {}
        }
        // Then fatally error, returning `!`
        crate::FatalError.raise()
    }

    fn make_diagnostic_builder(
        handler: &Handler,
        msg: impl Into<DiagnosticMessage>,
    ) -> DiagnosticBuilder<'_, Self> {
        DiagnosticBuilder::new_fatal(handler, msg)
    }
}

impl<'a> DiagnosticBuilder<'a, rustc_span::fatal_error::FatalError> {
    /// Convenience function for internal use, clients should use one of the
    /// `struct_*` methods on [`Handler`].
    #[track_caller]
    pub(crate) fn new_almost_fatal(
        handler: &'a Handler,
        message: impl Into<DiagnosticMessage>,
    ) -> Self {
        let diagnostic = Diagnostic::new_with_code(Level::Fatal, None, message);
        Self::new_diagnostic_almost_fatal(handler, diagnostic)
    }

    /// Creates a new `DiagnosticBuilder` with an already constructed
    /// diagnostic.
    pub(crate) fn new_diagnostic_almost_fatal(
        handler: &'a Handler,
        diagnostic: Diagnostic,
    ) -> Self {
        debug!("Created new diagnostic");
        Self {
            inner: DiagnosticBuilderInner {
                state: DiagnosticBuilderState::Emittable(handler),
                diagnostic: Box::new(diagnostic),
            },
            _marker: PhantomData,
        }
    }
}

impl EmissionGuarantee for rustc_span::fatal_error::FatalError {
    fn diagnostic_builder_emit_producing_guarantee(db: &mut DiagnosticBuilder<'_, Self>) -> Self {
        match db.inner.state {
            // First `.emit()` call, the `&Handler` is still available.
            DiagnosticBuilderState::Emittable(handler) => {
                db.inner.state = DiagnosticBuilderState::AlreadyEmittedOrDuringCancellation;

                handler.emit_diagnostic(&mut db.inner.diagnostic);
            }
            // `.emit()` was previously called, disallowed from repeating it.
            DiagnosticBuilderState::AlreadyEmittedOrDuringCancellation => {}
        }
        // Then fatally error..
        rustc_span::fatal_error::FatalError
    }

    fn make_diagnostic_builder(
        handler: &Handler,
        msg: impl Into<DiagnosticMessage>,
    ) -> DiagnosticBuilder<'_, Self> {
        DiagnosticBuilder::new_almost_fatal(handler, msg)
    }
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
    // Forward pattern for &mut self -> &mut Self
    (
        $(#[$attrs:meta])*
        pub fn $n:ident(&mut self, $($name:ident: $ty:ty),* $(,)?) -> &mut Self
    ) => {
        $(#[$attrs])*
        #[doc = concat!("See [`Diagnostic::", stringify!($n), "()`].")]
        pub fn $n(&mut self, $($name: $ty),*) -> &mut Self {
            self.inner.diagnostic.$n($($name),*);
            self
        }
    };
}

impl<G: EmissionGuarantee> Deref for DiagnosticBuilder<'_, G> {
    type Target = Diagnostic;

    fn deref(&self) -> &Diagnostic {
        &self.inner.diagnostic
    }
}

impl<G: EmissionGuarantee> DerefMut for DiagnosticBuilder<'_, G> {
    fn deref_mut(&mut self) -> &mut Diagnostic {
        &mut self.inner.diagnostic
    }
}

impl<'a, G: EmissionGuarantee> DiagnosticBuilder<'a, G> {
    /// Emit the diagnostic.
    #[track_caller]
    pub fn emit(&mut self) -> G {
        G::diagnostic_builder_emit_producing_guarantee(self)
    }

    /// Emit the diagnostic unless `delay` is true,
    /// in which case the emission will be delayed as a bug.
    ///
    /// See `emit` and `delay_as_bug` for details.
    #[track_caller]
    pub fn emit_unless(&mut self, delay: bool) -> G {
        if delay {
            self.downgrade_to_delayed_bug();
        }
        self.emit()
    }

    /// Cancel the diagnostic (a structured diagnostic must either be emitted or
    /// cancelled or it will panic when dropped).
    ///
    /// This method takes `self` by-value to disallow calling `.emit()` on it,
    /// which may be expected to *guarantee* the emission of an error, either
    /// at the time of the call, or through a prior `.emit()` call.
    pub fn cancel(mut self) {
        self.inner.state = DiagnosticBuilderState::AlreadyEmittedOrDuringCancellation;
        drop(self);
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
    /// unless handler has disabled such buffering, or `.emit()` was called.
    pub fn into_diagnostic(mut self) -> Option<(Diagnostic, &'a Handler)> {
        let handler = match self.inner.state {
            // No `.emit()` calls, the `&Handler` is still available.
            DiagnosticBuilderState::Emittable(handler) => handler,
            // `.emit()` was previously called, nothing we can do.
            DiagnosticBuilderState::AlreadyEmittedOrDuringCancellation => {
                return None;
            }
        };

        if handler.flags.dont_buffer_diagnostics || handler.flags.treat_err_as_bug.is_some() {
            self.emit();
            return None;
        }

        // Take the `Diagnostic` by replacing it with a dummy.
        let dummy = Diagnostic::new(Level::Allow, DiagnosticMessage::Str("".to_string()));
        let diagnostic = std::mem::replace(&mut *self.inner.diagnostic, dummy);

        // Disable the ICE on `Drop`.
        self.cancel();

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
    #[track_caller]
    pub fn delay_as_bug(&mut self) -> G {
        self.downgrade_to_delayed_bug();
        self.emit()
    }

    forward!(
        #[track_caller]
        pub fn downgrade_to_delayed_bug(&mut self,) -> &mut Self
    );

    forward!(
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
    pub fn span_label(&mut self, span: Span, label: impl Into<SubdiagnosticMessage>) -> &mut Self);

    forward!(
    /// Labels all the given spans with the provided label.
    /// See [`Diagnostic::span_label()`] for more information.
    pub fn span_labels(
        &mut self,
        spans: impl IntoIterator<Item = Span>,
        label: impl AsRef<str>,
    ) -> &mut Self);

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

    forward!(pub fn note(&mut self, msg: impl Into<SubdiagnosticMessage>) -> &mut Self);
    forward!(pub fn note_once(&mut self, msg: impl Into<SubdiagnosticMessage>) -> &mut Self);
    forward!(pub fn span_note(
        &mut self,
        sp: impl Into<MultiSpan>,
        msg: impl Into<SubdiagnosticMessage>,
    ) -> &mut Self);
    forward!(pub fn span_note_once(
        &mut self,
        sp: impl Into<MultiSpan>,
        msg: impl Into<SubdiagnosticMessage>,
    ) -> &mut Self);
    forward!(pub fn warn(&mut self, msg: impl Into<SubdiagnosticMessage>) -> &mut Self);
    forward!(pub fn span_warn(
        &mut self,
        sp: impl Into<MultiSpan>,
        msg: impl Into<SubdiagnosticMessage>,
    ) -> &mut Self);
    forward!(pub fn help(&mut self, msg: impl Into<SubdiagnosticMessage>) -> &mut Self);
    forward!(pub fn span_help(
        &mut self,
        sp: impl Into<MultiSpan>,
        msg: impl Into<SubdiagnosticMessage>,
    ) -> &mut Self);
    forward!(pub fn help_use_latest_edition(&mut self,) -> &mut Self);
    forward!(pub fn set_is_lint(&mut self,) -> &mut Self);

    forward!(pub fn disable_suggestions(&mut self,) -> &mut Self);
    forward!(pub fn clear_suggestions(&mut self,) -> &mut Self);

    forward!(pub fn multipart_suggestion(
        &mut self,
        msg: impl Into<SubdiagnosticMessage>,
        suggestion: Vec<(Span, String)>,
        applicability: Applicability,
    ) -> &mut Self);
    forward!(pub fn multipart_suggestion_verbose(
        &mut self,
        msg: impl Into<SubdiagnosticMessage>,
        suggestion: Vec<(Span, String)>,
        applicability: Applicability,
    ) -> &mut Self);
    forward!(pub fn tool_only_multipart_suggestion(
        &mut self,
        msg: impl Into<SubdiagnosticMessage>,
        suggestion: Vec<(Span, String)>,
        applicability: Applicability,
    ) -> &mut Self);
    forward!(pub fn span_suggestion(
        &mut self,
        sp: Span,
        msg: impl Into<SubdiagnosticMessage>,
        suggestion: impl ToString,
        applicability: Applicability,
    ) -> &mut Self);
    forward!(pub fn span_suggestions(
        &mut self,
        sp: Span,
        msg: impl Into<SubdiagnosticMessage>,
        suggestions: impl IntoIterator<Item = String>,
        applicability: Applicability,
    ) -> &mut Self);
    forward!(pub fn multipart_suggestions(
        &mut self,
        msg: impl Into<SubdiagnosticMessage>,
        suggestions: impl IntoIterator<Item = Vec<(Span, String)>>,
        applicability: Applicability,
    ) -> &mut Self);
    forward!(pub fn span_suggestion_short(
        &mut self,
        sp: Span,
        msg: impl Into<SubdiagnosticMessage>,
        suggestion: impl ToString,
        applicability: Applicability,
    ) -> &mut Self);
    forward!(pub fn span_suggestion_verbose(
        &mut self,
        sp: Span,
        msg: impl Into<SubdiagnosticMessage>,
        suggestion: impl ToString,
        applicability: Applicability,
    ) -> &mut Self);
    forward!(pub fn span_suggestion_hidden(
        &mut self,
        sp: Span,
        msg: impl Into<SubdiagnosticMessage>,
        suggestion: impl ToString,
        applicability: Applicability,
    ) -> &mut Self);
    forward!(pub fn tool_only_span_suggestion(
        &mut self,
        sp: Span,
        msg: impl Into<SubdiagnosticMessage>,
        suggestion: impl ToString,
        applicability: Applicability,
    ) -> &mut Self);

    forward!(pub fn set_primary_message(&mut self, msg: impl Into<DiagnosticMessage>) -> &mut Self);
    forward!(pub fn set_span(&mut self, sp: impl Into<MultiSpan>) -> &mut Self);
    forward!(pub fn code(&mut self, s: DiagnosticId) -> &mut Self);
    forward!(pub fn set_arg(
        &mut self,
        name: impl Into<Cow<'static, str>>,
        arg: impl IntoDiagnosticArg,
    ) -> &mut Self);

    forward!(pub fn subdiagnostic(
        &mut self,
        subdiagnostic: impl crate::AddToDiagnostic
    ) -> &mut Self);
}

impl<G: EmissionGuarantee> Debug for DiagnosticBuilder<'_, G> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.diagnostic.fmt(f)
    }
}

/// Destructor bomb - a `DiagnosticBuilder` must be either emitted or cancelled
/// or we emit a bug.
impl Drop for DiagnosticBuilderInner<'_> {
    fn drop(&mut self) {
        match self.state {
            // No `.emit()` or `.cancel()` calls.
            DiagnosticBuilderState::Emittable(handler) => {
                if !panicking() {
                    handler.emit_diagnostic(&mut Diagnostic::new(
                        Level::Bug,
                        DiagnosticMessage::Str(
                            "the following error was constructed but not emitted".to_string(),
                        ),
                    ));
                    handler.emit_diagnostic(&mut self.diagnostic);
                    panic!("error was constructed but not emitted");
                }
            }
            // `.emit()` was previously called, or maybe we're during `.cancel()`.
            DiagnosticBuilderState::AlreadyEmittedOrDuringCancellation => {}
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
