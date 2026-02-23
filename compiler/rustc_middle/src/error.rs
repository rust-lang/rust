use std::path::Path;
use std::{fmt, io};

use rustc_errors::codes::*;
use rustc_errors::{DiagArgName, DiagArgValue, DiagMessage};
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_span::{Span, Symbol};

use crate::ty::{Instance, Ty};

#[derive(Diagnostic)]
#[diag("overflow while adding drop-check rules for `{$ty}`", code = E0320)]
#[note("overflowed on `{$overflow_ty}`")]
pub(crate) struct DropCheckOverflow<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub overflow_ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag("failed to write file {$path}: {$error}\"")]
pub(crate) struct FailedWritingFile<'a> {
    pub path: &'a Path,
    pub error: io::Error,
}

#[derive(Diagnostic)]
#[diag("concrete type differs from previous defining opaque type use")]
pub(crate) struct OpaqueHiddenTypeMismatch<'tcx> {
    pub self_ty: Ty<'tcx>,
    pub other_ty: Ty<'tcx>,
    #[primary_span]
    #[label("expected `{$self_ty}`, got `{$other_ty}`")]
    pub other_span: Span,
    #[subdiagnostic]
    pub sub: TypeMismatchReason,
}

#[derive(Diagnostic)]
#[diag("we don't support unions yet: '{$ty_name}'")]
pub struct UnsupportedUnion {
    pub ty_name: String,
}

// FIXME(autodiff): I should get used somewhere
#[derive(Diagnostic)]
#[diag("reading from a `Duplicated` const {$ty} is unsafe")]
pub struct AutodiffUnsafeInnerConstRef<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
}

#[derive(Subdiagnostic)]
pub enum TypeMismatchReason {
    #[label("this expression supplies two conflicting concrete types for the same opaque type")]
    ConflictType {
        #[primary_span]
        span: Span,
    },
    #[note("previous use here")]
    PreviousUse {
        #[primary_span]
        span: Span,
    },
}

#[derive(Diagnostic)]
#[diag("reached the recursion limit finding the struct tail for `{$ty}`")]
#[help(
    "consider increasing the recursion limit by adding a `#![recursion_limit = \"{$suggested_limit}\"]`"
)]
pub(crate) struct RecursionLimitReached<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub suggested_limit: rustc_hir::limit::Limit,
}

#[derive(Diagnostic)]
#[diag("constant evaluation of enum discriminant resulted in non-integer")]
pub(crate) struct ConstEvalNonIntError {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(
    "to use `strict_coherence` on this trait, the `with_negative_coherence` feature must be enabled"
)]
pub(crate) struct StrictCoherenceNeedsNegativeCoherence {
    #[primary_span]
    pub span: Span,
    #[label("due to this attribute")]
    pub attr_span: Span,
}

#[derive(Diagnostic)]
#[diag("requires `{$name}` lang_item")]
pub(crate) struct RequiresLangItem {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(
    "const parameter `{$ct}` is part of concrete type but not used in parameter list for the `impl Trait` type alias"
)]
pub(super) struct ConstNotUsedTraitAlias {
    pub ct: String,
    #[primary_span]
    pub span: Span,
}

pub struct CustomSubdiagnostic<'a> {
    pub msg: fn() -> DiagMessage,
    pub add_args: Box<dyn FnOnce(&mut dyn FnMut(DiagArgName, DiagArgValue)) + 'a>,
}

impl<'a> CustomSubdiagnostic<'a> {
    pub fn label(x: fn() -> DiagMessage) -> Self {
        Self::label_and_then(x, |_| {})
    }
    pub fn label_and_then<F: FnOnce(&mut dyn FnMut(DiagArgName, DiagArgValue)) + 'a>(
        msg: fn() -> DiagMessage,
        f: F,
    ) -> Self {
        Self { msg, add_args: Box::new(move |x| f(x)) }
    }
}

impl fmt::Debug for CustomSubdiagnostic<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CustomSubdiagnostic").finish_non_exhaustive()
    }
}

#[derive(Diagnostic)]
pub enum LayoutError<'tcx> {
    #[diag("the type `{$ty}` has an unknown layout")]
    Unknown { ty: Ty<'tcx> },

    #[diag("the type `{$ty}` does not have a fixed layout")]
    TooGeneric { ty: Ty<'tcx> },

    #[diag("values of the type `{$ty}` are too big for the target architecture")]
    Overflow { ty: Ty<'tcx> },

    #[diag("the SIMD type `{$ty}` has more elements than the limit {$max_lanes}")]
    SimdTooManyLanes { ty: Ty<'tcx>, max_lanes: u64 },

    #[diag("the SIMD type `{$ty}` has zero elements")]
    SimdZeroLength { ty: Ty<'tcx> },

    #[diag("unable to determine layout for `{$ty}` because `{$failure_ty}` cannot be normalized")]
    NormalizationFailure { ty: Ty<'tcx>, failure_ty: String },

    #[diag("a cycle occurred during layout computation")]
    Cycle,

    #[diag("the type has an unknown layout")]
    ReferencesError,
}

#[derive(Diagnostic)]
#[diag("erroneous constant encountered")]
pub(crate) struct ErroneousConstant {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("reached the type-length limit while instantiating `{$instance}`")]
#[help("consider adding a `#![type_length_limit=\"{$type_length}\"]` attribute to your crate")]
pub(crate) struct TypeLengthLimit<'tcx> {
    #[primary_span]
    pub span: Span,
    pub instance: Instance<'tcx>,
    pub type_length: usize,
}

#[derive(Diagnostic)]
#[diag("maximum number of nodes exceeded in constant {$global_const_id}")]
pub(crate) struct MaxNumNodesInValtree {
    #[primary_span]
    pub span: Span,
    pub global_const_id: String,
}

#[derive(Diagnostic)]
#[diag("constant {$global_const_id} cannot be used as pattern")]
#[note("constants that reference mutable or external memory cannot be used as patterns")]
pub(crate) struct InvalidConstInValtree {
    #[primary_span]
    pub span: Span,
    pub global_const_id: String,
}

#[derive(Diagnostic)]
#[diag("internal compiler error: reentrant incremental verify failure, suppressing message")]
pub(crate) struct Reentrant;

#[derive(Diagnostic)]
#[diag("internal compiler error: encountered incremental compilation error with {$dep_node}")]
#[note("please follow the instructions below to create a bug report with the provided information")]
#[note("for incremental compilation bugs, having a reproduction is vital")]
#[note(
    "an ideal reproduction consists of the code before and some patch that then triggers the bug when applied and compiled again"
)]
#[note("as a workaround, you can run {$run_cmd} to allow your project to compile")]
pub(crate) struct IncrementCompilation {
    pub run_cmd: String,
    pub dep_node: String,
}
