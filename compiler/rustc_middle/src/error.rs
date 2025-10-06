use std::path::Path;
use std::{fmt, io};

use rustc_errors::codes::*;
use rustc_errors::{DiagArgName, DiagArgValue, DiagMessage};
use rustc_macros::{Diagnostic, Subdiagnostic};
use rustc_span::{Span, Symbol};

use crate::ty::{Instance, Ty};

#[derive(Diagnostic)]
#[diag(middle_drop_check_overflow, code = E0320)]
#[note]
pub(crate) struct DropCheckOverflow<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub overflow_ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag(middle_failed_writing_file)]
pub(crate) struct FailedWritingFile<'a> {
    pub path: &'a Path,
    pub error: io::Error,
}

#[derive(Diagnostic)]
#[diag(middle_opaque_hidden_type_mismatch)]
pub(crate) struct OpaqueHiddenTypeMismatch<'tcx> {
    pub self_ty: Ty<'tcx>,
    pub other_ty: Ty<'tcx>,
    #[primary_span]
    #[label]
    pub other_span: Span,
    #[subdiagnostic]
    pub sub: TypeMismatchReason,
}

#[derive(Diagnostic)]
#[diag(middle_unsupported_union)]
pub struct UnsupportedUnion {
    pub ty_name: String,
}

// FIXME(autodiff): I should get used somewhere
#[derive(Diagnostic)]
#[diag(middle_autodiff_unsafe_inner_const_ref)]
pub struct AutodiffUnsafeInnerConstRef<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
}

#[derive(Subdiagnostic)]
pub enum TypeMismatchReason {
    #[label(middle_conflict_types)]
    ConflictType {
        #[primary_span]
        span: Span,
    },
    #[note(middle_previous_use_here)]
    PreviousUse {
        #[primary_span]
        span: Span,
    },
}

#[derive(Diagnostic)]
#[diag(middle_recursion_limit_reached)]
#[help]
pub(crate) struct RecursionLimitReached<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub suggested_limit: rustc_hir::limit::Limit,
}

#[derive(Diagnostic)]
#[diag(middle_const_eval_non_int)]
pub(crate) struct ConstEvalNonIntError {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(middle_strict_coherence_needs_negative_coherence)]
pub(crate) struct StrictCoherenceNeedsNegativeCoherence {
    #[primary_span]
    pub span: Span,
    #[label]
    pub attr_span: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(middle_requires_lang_item)]
pub(crate) struct RequiresLangItem {
    #[primary_span]
    pub span: Span,
    pub name: Symbol,
}

#[derive(Diagnostic)]
#[diag(middle_const_not_used_in_type_alias)]
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
    #[diag(middle_layout_unknown)]
    Unknown { ty: Ty<'tcx> },

    #[diag(middle_layout_too_generic)]
    TooGeneric { ty: Ty<'tcx> },

    #[diag(middle_layout_size_overflow)]
    Overflow { ty: Ty<'tcx> },

    #[diag(middle_layout_simd_too_many)]
    SimdTooManyLanes { ty: Ty<'tcx>, max_lanes: u64 },

    #[diag(middle_layout_simd_zero_length)]
    SimdZeroLength { ty: Ty<'tcx> },

    #[diag(middle_layout_normalization_failure)]
    NormalizationFailure { ty: Ty<'tcx>, failure_ty: String },

    #[diag(middle_layout_cycle)]
    Cycle,

    #[diag(middle_layout_references_error)]
    ReferencesError,
}

#[derive(Diagnostic)]
#[diag(middle_erroneous_constant)]
pub(crate) struct ErroneousConstant {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(middle_type_length_limit)]
#[help(middle_consider_type_length_limit)]
pub(crate) struct TypeLengthLimit<'tcx> {
    #[primary_span]
    pub span: Span,
    pub instance: Instance<'tcx>,
    pub type_length: usize,
}

#[derive(Diagnostic)]
#[diag(middle_max_num_nodes_in_valtree)]
pub(crate) struct MaxNumNodesInValtree {
    #[primary_span]
    pub span: Span,
    pub global_const_id: String,
}

#[derive(Diagnostic)]
#[diag(middle_invalid_const_in_valtree)]
#[note]
pub(crate) struct InvalidConstInValtree {
    #[primary_span]
    pub span: Span,
    pub global_const_id: String,
}
