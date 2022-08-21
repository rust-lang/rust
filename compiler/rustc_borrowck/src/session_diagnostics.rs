use rustc_macros::{SessionDiagnostic, SessionSubdiagnostic};
use rustc_middle::ty::Ty;
use rustc_span::Span;

#[derive(SessionDiagnostic)]
#[diag(borrowck::move_unsized, code = "E0161")]
pub(crate) struct MoveUnsized<'tcx> {
    pub ty: Ty<'tcx>,
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(borrowck::higher_ranked_lifetime_error)]
pub(crate) struct HigherRankedLifetimeError {
    #[subdiagnostic]
    pub cause: Option<HigherRankedErrorCause>,
    #[primary_span]
    pub span: Span,
}

#[derive(SessionSubdiagnostic)]
pub(crate) enum HigherRankedErrorCause {
    #[note(borrowck::could_not_prove)]
    CouldNotProve { predicate: String },
    #[note(borrowck::could_not_normalize)]
    CouldNotNormalize { value: String },
}

#[derive(SessionDiagnostic)]
#[diag(borrowck::higher_ranked_subtype_error)]
pub(crate) struct HigherRankedSubtypeError {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(borrowck::generic_does_not_live_long_enough)]
pub(crate) struct GenericDoesNotLiveLongEnough {
    pub kind: String,
    #[primary_span]
    pub span: Span,
}
