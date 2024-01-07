use crate::diagnostic::IntoDiagnosticArg;
use crate::{DiagCtxt, Level, MultiSpan, StashKey};
use crate::{
    Diagnostic, DiagnosticId, DiagnosticMessage, DiagnosticStyledString, ErrorGuaranteed,
    ExplicitBug, SubdiagnosticMessage,
};
use rustc_lint_defs::Applicability;
use rustc_span::source_map::Spanned;

use rustc_span::Span;
use std::borrow::Cow;
use std::fmt::{self, Debug};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::panic;
use std::thread::panicking;

/// Trait implemented by error types. This is rarely implemented manually. Instead, use
/// `#[derive(Diagnostic)]` -- see [rustc_macros::Diagnostic].
#[rustc_diagnostic_item = "IntoDiagnostic"]
pub trait IntoDiagnostic<'a, G: EmissionGuarantee = ErrorGuaranteed> {
    /// Write out as a diagnostic out of `DiagCtxt`.
    #[must_use]
    fn into_diagnostic(self, dcx: &'a DiagCtxt, level: Level) -> DiagnosticBuilder<'a, G>;
}

impl<'a, T, G> IntoDiagnostic<'a, G> for Spanned<T>
where
    T: IntoDiagnostic<'a, G>,
    G: EmissionGuarantee,
{
    fn into_diagnostic(self, dcx: &'a DiagCtxt, level: Level) -> DiagnosticBuilder<'a, G> {
        let mut diag = self.node.into_diagnostic(dcx, level);
        diag.span(self.span);
        diag
    }
}

/// Used for emitting structured error messages and other diagnostic information.
///
/// If there is some state in a downstream crate you would like to
/// access in the methods of `DiagnosticBuilder` here, consider
/// extending `DiagCtxtFlags`.
#[must_use]
#[derive(Clone)]
pub struct DiagnosticBuilder<'a, G: EmissionGuarantee = ErrorGuaranteed> {
    state: DiagnosticBuilderState<'a>,

    /// `Diagnostic` is a large type, and `DiagnosticBuilder` is often used as a
    /// return value, especially within the frequently-used `PResult` type.
    /// In theory, return value optimization (RVO) should avoid unnecessary
    /// copying. In practice, it does not (at the time of writing).
    diagnostic: Box<Diagnostic>,

    _marker: PhantomData<G>,
}

#[derive(Clone)]
enum DiagnosticBuilderState<'a> {
    /// Initial state of a `DiagnosticBuilder`, before `.emit()` or `.cancel()`.
    ///
    /// The `Diagnostic` will be emitted through this `DiagCtxt`.
    Emittable(&'a DiagCtxt),

    /// State of a `DiagnosticBuilder`, after `.emit()` or *during* `.cancel()`.
    ///
    /// The `Diagnostic` will be ignored when calling `.emit()`, and it can be
    /// assumed that `.emit()` was previously called, to end up in this state.
    ///
    /// While this is also used by `.cancel()`, this state is only observed by
    /// the `Drop` `impl` of `DiagnosticBuilder`, because `.cancel()` takes
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
    std::mem::size_of::<&DiagCtxt>()
);

/// Trait for types that `DiagnosticBuilder::emit` can return as a "guarantee"
/// (or "proof") token that the emission happened.
pub trait EmissionGuarantee: Sized {
    /// This exists so that bugs and fatal errors can both result in `!` (an
    /// abort) when emitted, but have different aborting behaviour.
    type EmitResult = Self;

    /// Implementation of `DiagnosticBuilder::emit`, fully controlled by each
    /// `impl` of `EmissionGuarantee`, to make it impossible to create a value
    /// of `Self::EmitResult` without actually performing the emission.
    #[track_caller]
    fn emit_producing_guarantee(db: &mut DiagnosticBuilder<'_, Self>) -> Self::EmitResult;
}

impl<'a, G: EmissionGuarantee> DiagnosticBuilder<'a, G> {
    /// Most `emit_producing_guarantee` functions use this as a starting point.
    fn emit_producing_nothing(&mut self) {
        match self.state {
            // First `.emit()` call, the `&DiagCtxt` is still available.
            DiagnosticBuilderState::Emittable(dcx) => {
                self.state = DiagnosticBuilderState::AlreadyEmittedOrDuringCancellation;
                dcx.emit_diagnostic_without_consuming(&mut self.diagnostic);
            }
            // `.emit()` was previously called, disallowed from repeating it.
            DiagnosticBuilderState::AlreadyEmittedOrDuringCancellation => {}
        }
    }
}

// FIXME(eddyb) make `ErrorGuaranteed` impossible to create outside `.emit()`.
impl EmissionGuarantee for ErrorGuaranteed {
    fn emit_producing_guarantee(db: &mut DiagnosticBuilder<'_, Self>) -> Self::EmitResult {
        // Contrast this with `emit_producing_nothing`.
        match db.state {
            // First `.emit()` call, the `&DiagCtxt` is still available.
            DiagnosticBuilderState::Emittable(dcx) => {
                db.state = DiagnosticBuilderState::AlreadyEmittedOrDuringCancellation;
                let guar = dcx.emit_diagnostic_without_consuming(&mut db.diagnostic);

                // Only allow a guarantee if the `level` wasn't switched to a
                // non-error - the field isn't `pub`, but the whole `Diagnostic`
                // can be overwritten with a new one, thanks to `DerefMut`.
                assert!(
                    db.diagnostic.is_error(),
                    "emitted non-error ({:?}) diagnostic \
                     from `DiagnosticBuilder<ErrorGuaranteed>`",
                    db.diagnostic.level,
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
                    db.diagnostic.is_error(),
                    "`DiagnosticBuilder<ErrorGuaranteed>`'s diagnostic \
                     became non-error ({:?}), after original `.emit()`",
                    db.diagnostic.level,
                );
                #[allow(deprecated)]
                ErrorGuaranteed::unchecked_claim_error_was_emitted()
            }
        }
    }
}

// FIXME(eddyb) should there be a `Option<ErrorGuaranteed>` impl as well?
impl EmissionGuarantee for () {
    fn emit_producing_guarantee(db: &mut DiagnosticBuilder<'_, Self>) -> Self::EmitResult {
        db.emit_producing_nothing();
    }
}

/// Marker type which enables implementation of `create_bug` and `emit_bug` functions for
/// bug diagnostics.
#[derive(Copy, Clone)]
pub struct BugAbort;

impl EmissionGuarantee for BugAbort {
    type EmitResult = !;

    fn emit_producing_guarantee(db: &mut DiagnosticBuilder<'_, Self>) -> Self::EmitResult {
        db.emit_producing_nothing();
        panic::panic_any(ExplicitBug);
    }
}

/// Marker type which enables implementation of `create_fatal` and `emit_fatal` functions for
/// fatal diagnostics.
#[derive(Copy, Clone)]
pub struct FatalAbort;

impl EmissionGuarantee for FatalAbort {
    type EmitResult = !;

    fn emit_producing_guarantee(db: &mut DiagnosticBuilder<'_, Self>) -> Self::EmitResult {
        db.emit_producing_nothing();
        crate::FatalError.raise()
    }
}

impl EmissionGuarantee for rustc_span::fatal_error::FatalError {
    fn emit_producing_guarantee(db: &mut DiagnosticBuilder<'_, Self>) -> Self::EmitResult {
        db.emit_producing_nothing();
        rustc_span::fatal_error::FatalError
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
        pub fn $n:ident(&mut self $(, $name:ident: $ty:ty)* $(,)?) -> &mut Self
    ) => {
        $(#[$attrs])*
        #[doc = concat!("See [`Diagnostic::", stringify!($n), "()`].")]
        pub fn $n(&mut self $(, $name: $ty)*) -> &mut Self {
            self.diagnostic.$n($($name),*);
            self
        }
    };
}

impl<G: EmissionGuarantee> Deref for DiagnosticBuilder<'_, G> {
    type Target = Diagnostic;

    fn deref(&self) -> &Diagnostic {
        &self.diagnostic
    }
}

impl<G: EmissionGuarantee> DerefMut for DiagnosticBuilder<'_, G> {
    fn deref_mut(&mut self) -> &mut Diagnostic {
        &mut self.diagnostic
    }
}

impl<'a, G: EmissionGuarantee> DiagnosticBuilder<'a, G> {
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn new<M: Into<DiagnosticMessage>>(dcx: &'a DiagCtxt, level: Level, message: M) -> Self {
        Self::new_diagnostic(dcx, Diagnostic::new(level, message))
    }

    /// Creates a new `DiagnosticBuilder` with an already constructed
    /// diagnostic.
    #[track_caller]
    pub(crate) fn new_diagnostic(dcx: &'a DiagCtxt, diagnostic: Diagnostic) -> Self {
        debug!("Created new diagnostic");
        Self {
            state: DiagnosticBuilderState::Emittable(dcx),
            diagnostic: Box::new(diagnostic),
            _marker: PhantomData,
        }
    }

    /// Emit the diagnostic. Does not consume `self`, which may be surprising,
    /// but there are various places that rely on continuing to use `self`
    /// after calling `emit`.
    #[track_caller]
    pub fn emit(&mut self) -> G::EmitResult {
        G::emit_producing_guarantee(self)
    }

    /// Emit the diagnostic unless `delay` is true,
    /// in which case the emission will be delayed as a bug.
    ///
    /// See `emit` and `delay_as_bug` for details.
    #[track_caller]
    pub fn emit_unless(&mut self, delay: bool) -> G::EmitResult {
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
        self.state = DiagnosticBuilderState::AlreadyEmittedOrDuringCancellation;
        drop(self);
    }

    /// Stashes diagnostic for possible later improvement in a different,
    /// later stage of the compiler. The diagnostic can be accessed with
    /// the provided `span` and `key` through [`DiagCtxt::steal_diagnostic()`].
    ///
    /// As with `buffer`, this is unless the dcx has disabled such buffering.
    pub fn stash(self, span: Span, key: StashKey) {
        if let Some((diag, dcx)) = self.into_diagnostic() {
            dcx.stash_diagnostic(span, key, diag);
        }
    }

    /// Converts the builder to a `Diagnostic` for later emission,
    /// unless dcx has disabled such buffering, or `.emit()` was called.
    pub fn into_diagnostic(mut self) -> Option<(Diagnostic, &'a DiagCtxt)> {
        let dcx = match self.state {
            // No `.emit()` calls, the `&DiagCtxt` is still available.
            DiagnosticBuilderState::Emittable(dcx) => dcx,
            // `.emit()` was previously called, nothing we can do.
            DiagnosticBuilderState::AlreadyEmittedOrDuringCancellation => {
                return None;
            }
        };

        if dcx.inner.lock().flags.treat_err_as_bug.is_some() {
            self.emit();
            return None;
        }

        // Take the `Diagnostic` by replacing it with a dummy.
        let dummy = Diagnostic::new(Level::Allow, DiagnosticMessage::from(""));
        let diagnostic = std::mem::replace(&mut *self.diagnostic, dummy);

        // Disable the ICE on `Drop`.
        self.cancel();

        // Logging here is useful to help track down where in logs an error was
        // actually emitted.
        debug!("buffer: diagnostic={:?}", diagnostic);

        Some((diagnostic, dcx))
    }

    /// Retrieves the [`DiagCtxt`] if available
    pub fn dcx(&self) -> Option<&DiagCtxt> {
        match self.state {
            DiagnosticBuilderState::Emittable(dcx) => Some(dcx),
            DiagnosticBuilderState::AlreadyEmittedOrDuringCancellation => None,
        }
    }

    /// Buffers the diagnostic for later emission,
    /// unless dcx has disabled such buffering.
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
    pub fn delay_as_bug(&mut self) -> G::EmitResult {
        self.downgrade_to_delayed_bug();
        self.emit()
    }

    forward!(pub fn span_label(
        &mut self,
        span: Span,
        label: impl Into<SubdiagnosticMessage>
    ) -> &mut Self);
    forward!(pub fn span_labels(
        &mut self,
        spans: impl IntoIterator<Item = Span>,
        label: &str,
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
    forward!(pub fn help_once(&mut self, msg: impl Into<SubdiagnosticMessage>) -> &mut Self);
    forward!(pub fn span_help(
        &mut self,
        sp: impl Into<MultiSpan>,
        msg: impl Into<SubdiagnosticMessage>,
    ) -> &mut Self);
    forward!(pub fn is_lint(&mut self) -> &mut Self);
    forward!(pub fn disable_suggestions(&mut self) -> &mut Self);
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
    forward!(pub fn primary_message(&mut self, msg: impl Into<DiagnosticMessage>) -> &mut Self);
    forward!(pub fn span(&mut self, sp: impl Into<MultiSpan>) -> &mut Self);
    forward!(pub fn code(&mut self, s: DiagnosticId) -> &mut Self);
    forward!(pub fn arg(
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
        self.diagnostic.fmt(f)
    }
}

/// Destructor bomb - a `DiagnosticBuilder` must be either emitted or cancelled
/// or we emit a bug.
impl<G: EmissionGuarantee> Drop for DiagnosticBuilder<'_, G> {
    fn drop(&mut self) {
        match self.state {
            // No `.emit()` or `.cancel()` calls.
            DiagnosticBuilderState::Emittable(dcx) => {
                if !panicking() {
                    dcx.emit_diagnostic(Diagnostic::new(
                        Level::Bug,
                        DiagnosticMessage::from(
                            "the following error was constructed but not emitted",
                        ),
                    ));
                    dcx.emit_diagnostic_without_consuming(&mut self.diagnostic);
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
    ($dcx:expr, $span:expr, $code:ident, $($message:tt)*) => ({
        $dcx.struct_span_err_with_code(
            $span,
            format!($($message)*),
            $crate::error_code!($code),
        )
    })
}

#[macro_export]
macro_rules! error_code {
    ($code:ident) => {{ $crate::DiagnosticId::Error(stringify!($code).to_owned()) }};
}
