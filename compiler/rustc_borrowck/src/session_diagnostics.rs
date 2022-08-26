use rustc_errors::{IntoDiagnosticArg, MultiSpan};
use rustc_macros::{LintDiagnostic, SessionDiagnostic, SessionSubdiagnostic};
use rustc_middle::ty::Ty;
use rustc_span::Span;

use crate::diagnostics::RegionName;

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

#[derive(LintDiagnostic)]
#[diag(borrowck::var_does_not_need_mut)]
pub(crate) struct VarNeedNotMut {
    #[suggestion_short(applicability = "machine-applicable", code = "")]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(borrowck::const_not_used_in_type_alias)]
pub(crate) struct ConstNotUsedTraitAlias {
    pub ct: String,
    #[primary_span]
    pub span: Span,
}

#[derive(SessionDiagnostic)]
#[diag(borrowck::var_cannot_escape_closure)]
#[note]
#[note(borrowck::cannot_escape)]
pub(crate) struct FnMutError {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub ty_err: FnMutReturnTypeErr,
}

#[derive(SessionSubdiagnostic)]
pub(crate) enum VarHereDenote {
    #[label(borrowck::var_here_captured)]
    Captured {
        #[primary_span]
        span: Span,
    },
    #[label(borrowck::var_here_defined)]
    Defined {
        #[primary_span]
        span: Span,
    },
    #[label(borrowck::closure_inferred_mut)]
    FnMutInferred {
        #[primary_span]
        span: Span,
    },
}

#[derive(SessionSubdiagnostic)]
pub(crate) enum FnMutReturnTypeErr {
    #[label(borrowck::returned_closure_escaped)]
    ReturnClosure {
        #[primary_span]
        span: Span,
    },
    #[label(borrowck::returned_async_block_escaped)]
    ReturnAsyncBlock {
        #[primary_span]
        span: Span,
    },
    #[label(borrowck::returned_ref_escaped)]
    ReturnRef {
        #[primary_span]
        span: Span,
    },
}

#[derive(SessionDiagnostic)]
#[diag(borrowck::lifetime_constraints_error)]
pub(crate) struct LifetimeOutliveErr {
    #[primary_span]
    pub span: Span,
}

#[derive(SessionSubdiagnostic)]
pub(crate) enum LifetimeReturnCategoryErr<'a> {
    #[label(borrowck::returned_lifetime_wrong)]
    WrongReturn {
        #[primary_span]
        span: Span,
        mir_def_name: &'a str,
        outlived_fr_name: RegionName,
        fr_name: &'a RegionName,
    },
    #[label(borrowck::returned_lifetime_short)]
    ShortReturn {
        #[primary_span]
        span: Span,
        category_desc: &'static str,
        free_region_name: &'a RegionName,
        outlived_fr_name: RegionName,
    },
}

impl IntoDiagnosticArg for &RegionName {
    fn into_diagnostic_arg(self) -> rustc_errors::DiagnosticArgValue<'static> {
        format!("{}", self).into_diagnostic_arg()
    }
}

impl IntoDiagnosticArg for RegionName {
    fn into_diagnostic_arg(self) -> rustc_errors::DiagnosticArgValue<'static> {
        format!("{}", self).into_diagnostic_arg()
    }
}

#[derive(SessionSubdiagnostic)]
pub(crate) enum RequireStaticErr {
    #[note(borrowck::used_impl_require_static)]
    UsedImpl {
        #[primary_span]
        multi_span: MultiSpan,
    },
}
