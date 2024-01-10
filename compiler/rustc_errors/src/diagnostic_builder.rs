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
        self.node.into_diagnostic(dcx, level).span_mv(self.span)
    }
}

/// Used for emitting structured error messages and other diagnostic information.
/// Each constructed `DiagnosticBuilder` must be consumed by a function such as
/// `emit`, `cancel`, `delay_as_bug`, or `into_diagnostic`. A panic occurrs if a
/// `DiagnosticBuilder` is dropped without being consumed by one of these
/// functions.
///
/// If there is some state in a downstream crate you would like to
/// access in the methods of `DiagnosticBuilder` here, consider
/// extending `DiagCtxtFlags`.
#[must_use]
pub struct DiagnosticBuilder<'a, G: EmissionGuarantee = ErrorGuaranteed> {
    pub dcx: &'a DiagCtxt,

    /// Why the `Option`? It is always `Some` until the `DiagnosticBuilder` is
    /// consumed via `emit`, `cancel`, etc. At that point it is consumed and
    /// replaced with `None`. Then `drop` checks that it is `None`; if not, it
    /// panics because a diagnostic was built but not used.
    ///
    /// Why the Box? `Diagnostic` is a large type, and `DiagnosticBuilder` is
    /// often used as a return value, especially within the frequently-used
    /// `PResult` type. In theory, return value optimization (RVO) should avoid
    /// unnecessary copying. In practice, it does not (at the time of writing).
    diag: Option<Box<Diagnostic>>,

    _marker: PhantomData<G>,
}

// Cloning a `DiagnosticBuilder` is a recipe for a diagnostic being emitted
// twice, which would be bad.
impl<G> !Clone for DiagnosticBuilder<'_, G> {}

rustc_data_structures::static_assert_size!(
    DiagnosticBuilder<'_, ()>,
    2 * std::mem::size_of::<usize>()
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
    fn emit_producing_guarantee(db: DiagnosticBuilder<'_, Self>) -> Self::EmitResult;
}

impl<'a, G: EmissionGuarantee> DiagnosticBuilder<'a, G> {
    /// Takes the diagnostic. For use by methods that consume the
    /// DiagnosticBuilder: `emit`, `cancel`, etc. Afterwards, `drop` is the
    /// only code that will be run on `self`.
    fn take_diag(&mut self) -> Diagnostic {
        Box::into_inner(self.diag.take().unwrap())
    }

    /// Most `emit_producing_guarantee` functions use this as a starting point.
    fn emit_producing_nothing(mut self) {
        let diag = self.take_diag();
        self.dcx.emit_diagnostic(diag);
    }

    /// `ErrorGuaranteed::emit_producing_guarantee` uses this.
    // FIXME(eddyb) make `ErrorGuaranteed` impossible to create outside `.emit()`.
    fn emit_producing_error_guaranteed(mut self) -> ErrorGuaranteed {
        let diag = self.take_diag();

        // Only allow a guarantee if the `level` wasn't switched to a
        // non-error. The field isn't `pub`, but the whole `Diagnostic` can be
        // overwritten with a new one, thanks to `DerefMut`.
        assert!(
            diag.is_error(),
            "emitted non-error ({:?}) diagnostic from `DiagnosticBuilder<ErrorGuaranteed>`",
            diag.level,
        );

        let guar = self.dcx.emit_diagnostic(diag);
        guar.unwrap()
    }
}

impl EmissionGuarantee for ErrorGuaranteed {
    fn emit_producing_guarantee(db: DiagnosticBuilder<'_, Self>) -> Self::EmitResult {
        db.emit_producing_error_guaranteed()
    }
}

impl EmissionGuarantee for () {
    fn emit_producing_guarantee(db: DiagnosticBuilder<'_, Self>) -> Self::EmitResult {
        db.emit_producing_nothing();
    }
}

/// Marker type which enables implementation of `create_bug` and `emit_bug` functions for
/// bug diagnostics.
#[derive(Copy, Clone)]
pub struct BugAbort;

impl EmissionGuarantee for BugAbort {
    type EmitResult = !;

    fn emit_producing_guarantee(db: DiagnosticBuilder<'_, Self>) -> Self::EmitResult {
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

    fn emit_producing_guarantee(db: DiagnosticBuilder<'_, Self>) -> Self::EmitResult {
        db.emit_producing_nothing();
        crate::FatalError.raise()
    }
}

impl EmissionGuarantee for rustc_span::fatal_error::FatalError {
    fn emit_producing_guarantee(db: DiagnosticBuilder<'_, Self>) -> Self::EmitResult {
        db.emit_producing_nothing();
        rustc_span::fatal_error::FatalError
    }
}

/// `DiagnosticBuilder` impls `DerefMut`, which allows access to the fields and
/// methods of the embedded `Diagnostic`. However, that doesn't allow method
/// chaining at the `DiagnosticBuilder` level. Each use of this macro defines
/// two builder methods at that level, both of which wrap the equivalent method
/// in `Diagnostic`.
/// - A `&mut self -> &mut Self` method, with the same name as the underlying
///   `Diagnostic` method. It is mostly to modify existing diagnostics, either
///   in a standalone fashion, e.g. `err.code(code)`, or in a chained fashion
///   to make multiple modifications, e.g. `err.code(code).span(span)`.
/// - A `self -> Self` method, with `_mv` suffix added (short for "move").
///   It is mostly used in a chained fashion when producing a new diagnostic,
///   e.g. `let err = struct_err(msg).code_mv(code)`, or when emitting a new
///   diagnostic , e.g. `struct_err(msg).code_mv(code).emit()`.
///
/// Although the latter method can be used to modify an existing diagnostic,
/// e.g. `err = err.code_mv(code)`, this should be avoided because the former
/// method give shorter code, e.g. `err.code(code)`.
macro_rules! forward {
    (
        ($n:ident, $n_mv:ident)($($name:ident: $ty:ty),* $(,)?)
    ) => {
        #[doc = concat!("See [`Diagnostic::", stringify!($n), "()`].")]
        pub fn $n(&mut self, $($name: $ty),*) -> &mut Self {
            self.diag.as_mut().unwrap().$n($($name),*);
            self
        }
        #[doc = concat!("See [`Diagnostic::", stringify!($n), "()`].")]
        pub fn $n_mv(mut self, $($name: $ty),*) -> Self {
            self.diag.as_mut().unwrap().$n($($name),*);
            self
        }
    };
}

impl<G: EmissionGuarantee> Deref for DiagnosticBuilder<'_, G> {
    type Target = Diagnostic;

    fn deref(&self) -> &Diagnostic {
        self.diag.as_ref().unwrap()
    }
}

impl<G: EmissionGuarantee> DerefMut for DiagnosticBuilder<'_, G> {
    fn deref_mut(&mut self) -> &mut Diagnostic {
        self.diag.as_mut().unwrap()
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
    pub(crate) fn new_diagnostic(dcx: &'a DiagCtxt, diag: Diagnostic) -> Self {
        debug!("Created new diagnostic");
        Self { dcx, diag: Some(Box::new(diag)), _marker: PhantomData }
    }

    /// Emit and consume the diagnostic.
    #[track_caller]
    pub fn emit(self) -> G::EmitResult {
        G::emit_producing_guarantee(self)
    }

    /// Emit the diagnostic unless `delay` is true,
    /// in which case the emission will be delayed as a bug.
    ///
    /// See `emit` and `delay_as_bug` for details.
    #[track_caller]
    pub fn emit_unless(mut self, delay: bool) -> G::EmitResult {
        if delay {
            self.downgrade_to_delayed_bug();
        }
        self.emit()
    }

    /// Cancel and consume the diagnostic. (A diagnostic must either be emitted or
    /// cancelled or it will panic when dropped).
    pub fn cancel(mut self) {
        self.diag = None;
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
    /// unless dcx has disabled such buffering.
    pub fn into_diagnostic(mut self) -> Option<(Diagnostic, &'a DiagCtxt)> {
        if self.dcx.inner.lock().flags.treat_err_as_bug.is_some() {
            self.emit();
            return None;
        }

        let diag = self.take_diag();

        // Logging here is useful to help track down where in logs an error was
        // actually emitted.
        debug!("buffer: diag={:?}", diag);

        Some((diag, self.dcx))
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
    pub fn delay_as_bug(mut self) -> G::EmitResult {
        self.downgrade_to_delayed_bug();
        self.emit()
    }

    forward!((span_label, span_label_mv)(
        span: Span,
        label: impl Into<SubdiagnosticMessage>,
    ));
    forward!((span_labels, span_labels_mv)(
        spans: impl IntoIterator<Item = Span>,
        label: &str,
    ));
    forward!((note_expected_found, note_expected_found_mv)(
        expected_label: &dyn fmt::Display,
        expected: DiagnosticStyledString,
        found_label: &dyn fmt::Display,
        found: DiagnosticStyledString,
    ));
    forward!((note_expected_found_extra, note_expected_found_extra_mv)(
        expected_label: &dyn fmt::Display,
        expected: DiagnosticStyledString,
        found_label: &dyn fmt::Display,
        found: DiagnosticStyledString,
        expected_extra: &dyn fmt::Display,
        found_extra: &dyn fmt::Display,
    ));
    forward!((note, note_mv)(
        msg: impl Into<SubdiagnosticMessage>,
    ));
    forward!((note_once, note_once_mv)(
        msg: impl Into<SubdiagnosticMessage>,
    ));
    forward!((span_note, span_note_mv)(
        sp: impl Into<MultiSpan>,
        msg: impl Into<SubdiagnosticMessage>,
    ));
    forward!((span_note_once, span_note_once_mv)(
        sp: impl Into<MultiSpan>,
        msg: impl Into<SubdiagnosticMessage>,
    ));
    forward!((warn, warn_mv)(
        msg: impl Into<SubdiagnosticMessage>,
    ));
    forward!((span_warn, span_warn_mv)(
        sp: impl Into<MultiSpan>,
        msg: impl Into<SubdiagnosticMessage>,
    ));
    forward!((help, help_mv)(
        msg: impl Into<SubdiagnosticMessage>,
    ));
    forward!((help_once, help_once_mv)(
        msg: impl Into<SubdiagnosticMessage>,
    ));
    forward!((span_help, span_help_once_mv)(
        sp: impl Into<MultiSpan>,
        msg: impl Into<SubdiagnosticMessage>,
    ));
    forward!((multipart_suggestion, multipart_suggestion_mv)(
        msg: impl Into<SubdiagnosticMessage>,
        suggestion: Vec<(Span, String)>,
        applicability: Applicability,
    ));
    forward!((multipart_suggestion_verbose, multipart_suggestion_verbose_mv)(
        msg: impl Into<SubdiagnosticMessage>,
        suggestion: Vec<(Span, String)>,
        applicability: Applicability,
    ));
    forward!((tool_only_multipart_suggestion, tool_only_multipart_suggestion_mv)(
        msg: impl Into<SubdiagnosticMessage>,
        suggestion: Vec<(Span, String)>,
        applicability: Applicability,
    ));
    forward!((span_suggestion, span_suggestion_mv)(
        sp: Span,
        msg: impl Into<SubdiagnosticMessage>,
        suggestion: impl ToString,
        applicability: Applicability,
    ));
    forward!((span_suggestions, span_suggestions_mv)(
        sp: Span,
        msg: impl Into<SubdiagnosticMessage>,
        suggestions: impl IntoIterator<Item = String>,
        applicability: Applicability,
    ));
    forward!((multipart_suggestions, multipart_suggestions_mv)(
        msg: impl Into<SubdiagnosticMessage>,
        suggestions: impl IntoIterator<Item = Vec<(Span, String)>>,
        applicability: Applicability,
    ));
    forward!((span_suggestion_short, span_suggestion_short_mv)(
        sp: Span,
        msg: impl Into<SubdiagnosticMessage>,
        suggestion: impl ToString,
        applicability: Applicability,
    ));
    forward!((span_suggestion_verbose, span_suggestion_verbose_mv)(
        sp: Span,
        msg: impl Into<SubdiagnosticMessage>,
        suggestion: impl ToString,
        applicability: Applicability,
    ));
    forward!((span_suggestion_hidden, span_suggestion_hidden_mv)(
        sp: Span,
        msg: impl Into<SubdiagnosticMessage>,
        suggestion: impl ToString,
        applicability: Applicability,
    ));
    forward!((tool_only_span_suggestion, tool_only_span_suggestion_mv)(
        sp: Span,
        msg: impl Into<SubdiagnosticMessage>,
        suggestion: impl ToString,
        applicability: Applicability,
    ));
    forward!((primary_message, primary_message_mv)(
        msg: impl Into<DiagnosticMessage>,
    ));
    forward!((span, span_mv)(
        sp: impl Into<MultiSpan>,
    ));
    forward!((code, code_mv)(
        s: DiagnosticId,
    ));
    forward!((arg, arg_mv)(
        name: impl Into<Cow<'static, str>>, arg: impl IntoDiagnosticArg,
    ));
    forward!((subdiagnostic, subdiagnostic_mv)(
        subdiagnostic: impl crate::AddToDiagnostic,
    ));
}

impl<G: EmissionGuarantee> Debug for DiagnosticBuilder<'_, G> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.diag.fmt(f)
    }
}

/// Destructor bomb: every `DiagnosticBuilder` must be consumed (emitted,
/// cancelled, etc.) or we emit a bug.
impl<G: EmissionGuarantee> Drop for DiagnosticBuilder<'_, G> {
    fn drop(&mut self) {
        match self.diag.take() {
            Some(diag) if !panicking() => {
                self.dcx.emit_diagnostic(Diagnostic::new(
                    Level::Bug,
                    DiagnosticMessage::from("the following error was constructed but not emitted"),
                ));
                self.dcx.emit_diagnostic(*diag);
                panic!("error was constructed but not emitted");
            }
            _ => {}
        }
    }
}

#[macro_export]
macro_rules! struct_span_err {
    ($dcx:expr, $span:expr, $code:ident, $($message:tt)*) => ({
        $dcx.struct_span_err(
            $span,
            format!($($message)*),
        )
        .code_mv($crate::error_code!($code))
    })
}

#[macro_export]
macro_rules! error_code {
    ($code:ident) => {{ $crate::DiagnosticId::Error(stringify!($code).to_owned()) }};
}
