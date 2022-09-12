use rustc_macros::SessionDiagnostic;
use rustc_span::Span;

use crate::ty::Ty;

#[derive(SessionDiagnostic)]
#[diag(middle::drop_check_overflow, code = "E0320")]
#[note]
pub struct DropCheckOverflow<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub overflow_ty: Ty<'tcx>,
}

#[derive(SessionDiagnostic)]
#[diag(middle::opaque_hidden_type_mismatch)]
pub struct OpaqueHiddenTypeMismatch<'tcx> {
    pub self_ty: Ty<'tcx>,
    pub other_ty: Ty<'tcx>,
    #[primary_span]
    #[label]
    pub other_span: Span,
    #[subdiagnostic]
    pub sub: TypeMismatchReason,
}

#[derive(SessionSubdiagnostic)]
pub enum TypeMismatchReason {
    #[label(middle::conflict_types)]
    ConflictType {
        #[primary_span]
        span: Span,
    },
    #[note(middle::previous_use_here)]
    PreviousUse {
        #[primary_span]
        span: Span,
    },
}

#[derive(SessionDiagnostic)]
#[diag(middle::limit_invalid)]
pub struct LimitInvalid<'a> {
    #[primary_span]
    pub span: Span,
    #[label]
    pub value_span: Span,
    pub error_str: &'a str,
}
