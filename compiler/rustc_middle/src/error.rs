use std::borrow::Cow;
use std::fmt;

use rustc_errors::{DiagnosticArgValue, DiagnosticMessage};
use rustc_macros::Diagnostic;
use rustc_span::{Span, Symbol};

use crate::ty::Ty;

#[derive(Diagnostic)]
#[diag(middle_drop_check_overflow, code = "E0320")]
#[note]
pub struct DropCheckOverflow<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub overflow_ty: Ty<'tcx>,
}

#[derive(Diagnostic)]
#[diag(middle_opaque_hidden_type_mismatch)]
pub struct OpaqueHiddenTypeMismatch<'tcx> {
    pub self_ty: Ty<'tcx>,
    pub other_ty: Ty<'tcx>,
    #[primary_span]
    #[label]
    pub other_span: Span,
    #[subdiagnostic]
    pub sub: TypeMismatchReason,
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
#[diag(middle_limit_invalid)]
pub struct LimitInvalid<'a> {
    #[primary_span]
    pub span: Span,
    #[label]
    pub value_span: Span,
    pub error_str: &'a str,
}

#[derive(Diagnostic)]
#[diag(middle_recursion_limit_reached)]
#[help]
pub struct RecursionLimitReached<'tcx> {
    pub ty: Ty<'tcx>,
    pub suggested_limit: rustc_session::Limit,
}

#[derive(Diagnostic)]
#[diag(middle_const_eval_non_int)]
pub struct ConstEvalNonIntError {
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
    pub span: Option<Span>,
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
    pub msg: fn() -> DiagnosticMessage,
    pub add_args:
        Box<dyn FnOnce(&mut dyn FnMut(Cow<'static, str>, DiagnosticArgValue<'static>)) + 'a>,
}

impl<'a> CustomSubdiagnostic<'a> {
    pub fn label(x: fn() -> DiagnosticMessage) -> Self {
        Self::label_and_then(x, |_| {})
    }
    pub fn label_and_then<
        F: FnOnce(&mut dyn FnMut(Cow<'static, str>, DiagnosticArgValue<'static>)) + 'a,
    >(
        msg: fn() -> DiagnosticMessage,
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
    #[diag(middle_unknown_layout)]
    Unknown { ty: Ty<'tcx> },

    #[diag(middle_values_too_big)]
    Overflow { ty: Ty<'tcx> },

    #[diag(middle_cannot_be_normalized)]
    NormalizationFailure { ty: Ty<'tcx>, failure_ty: String },

    #[diag(middle_cycle)]
    Cycle,
}

#[derive(Diagnostic)]
#[diag(middle_adjust_for_foreign_abi_error)]
pub struct UnsupportedFnAbi {
    pub arch: Symbol,
    pub abi: &'static str,
}

/// Used by `rustc_const_eval`
pub use crate::fluent_generated::middle_adjust_for_foreign_abi_error;
