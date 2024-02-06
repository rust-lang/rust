use crate::{
    DiagCtxt, Diagnostic, DiagnosticMessage, ErrorGuaranteed, ExplicitBug, Level, StashKey,
};
use rustc_span::source_map::Spanned;
use rustc_span::Span;
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
        self.node.into_diagnostic(dcx, level).with_span(self.span)
    }
}

/// Used for emitting structured error messages and other diagnostic information.
/// Wraps a `Diagnostic`, adding some useful things.
/// - The `dcx` field, allowing it to (a) emit itself, and (b) do a drop check
///   that it has been emitted or cancelled.
/// - The `EmissionGuarantee`, which determines the type returned from `emit`.
///
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
    // FIXME(nnethercote) Make private once this moves to diagnostic.rs.
    pub(crate) diag: Option<Box<Diagnostic>>,

    // FIXME(nnethercote) Make private once this moves to diagnostic.rs.
    pub(crate) _marker: PhantomData<G>,
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
    // FIXME(nnethercote) Make private once this moves to diagnostic.rs.
    pub(crate) fn take_diag(&mut self) -> Diagnostic {
        Box::into_inner(self.diag.take().unwrap())
    }

    /// Most `emit_producing_guarantee` functions use this as a starting point.
    // FIXME(nnethercote) Make private once this moves to diagnostic.rs.
    pub(crate) fn emit_producing_nothing(mut self) {
        let diag = self.take_diag();
        self.dcx.emit_diagnostic(diag);
    }

    /// `ErrorGuaranteed::emit_producing_guarantee` uses this.
    // FIXME(nnethercote) Make private once this moves to diagnostic.rs.
    pub(crate) fn emit_producing_error_guaranteed(mut self) -> ErrorGuaranteed {
        let diag = self.take_diag();

        // The only error levels that produce `ErrorGuaranteed` are
        // `Error` and `DelayedBug`. But `DelayedBug` should never occur here
        // because delayed bugs have their level changed to `Bug` when they are
        // actually printed, so they produce an ICE.
        //
        // (Also, even though `level` isn't `pub`, the whole `Diagnostic` could
        // be overwritten with a new one thanks to `DerefMut`. So this assert
        // protects against that, too.)
        assert!(
            matches!(diag.level, Level::Error | Level::DelayedBug),
            "invalid diagnostic level ({:?})",
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
    pub fn stash(mut self, span: Span, key: StashKey) {
        self.dcx.stash_diagnostic(span, key, self.take_diag());
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
macro_rules! struct_span_code_err {
    ($dcx:expr, $span:expr, $code:expr, $($message:tt)*) => ({
        $dcx.struct_span_err($span, format!($($message)*)).with_code($code)
    })
}
