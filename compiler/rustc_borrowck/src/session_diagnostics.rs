use rustc_errors::MultiSpan;
use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};
use rustc_middle::ty::{GenericArg, Ty};
use rustc_span::Span;

use crate::diagnostics::RegionName;

#[derive(Diagnostic)]
#[diag(borrowck_move_unsized, code = "E0161")]
pub(crate) struct MoveUnsized<'tcx> {
    pub ty: Ty<'tcx>,
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(borrowck_higher_ranked_lifetime_error)]
pub(crate) struct HigherRankedLifetimeError {
    #[subdiagnostic]
    pub cause: Option<HigherRankedErrorCause>,
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
pub(crate) enum HigherRankedErrorCause {
    #[note(borrowck_could_not_prove)]
    CouldNotProve { predicate: String },
    #[note(borrowck_could_not_normalize)]
    CouldNotNormalize { value: String },
}

#[derive(Diagnostic)]
#[diag(borrowck_higher_ranked_subtype_error)]
pub(crate) struct HigherRankedSubtypeError {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(borrowck_generic_does_not_live_long_enough)]
pub(crate) struct GenericDoesNotLiveLongEnough {
    pub kind: String,
    #[primary_span]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag(borrowck_var_does_not_need_mut)]
pub(crate) struct VarNeedNotMut {
    #[suggestion(style = "short", applicability = "machine-applicable", code = "")]
    pub span: Span,
}
#[derive(Diagnostic)]
#[diag(borrowck_var_cannot_escape_closure)]
#[note]
#[note(cannot_escape)]
pub(crate) struct FnMutError {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub ty_err: FnMutReturnTypeErr,
}

#[derive(Subdiagnostic)]
pub(crate) enum VarHereDenote {
    #[label(borrowck_var_here_captured)]
    Captured {
        #[primary_span]
        span: Span,
    },
    #[label(borrowck_var_here_defined)]
    Defined {
        #[primary_span]
        span: Span,
    },
    #[label(borrowck_closure_inferred_mut)]
    FnMutInferred {
        #[primary_span]
        span: Span,
    },
}

#[derive(Subdiagnostic)]
pub(crate) enum FnMutReturnTypeErr {
    #[label(borrowck_returned_closure_escaped)]
    ReturnClosure {
        #[primary_span]
        span: Span,
    },
    #[label(borrowck_returned_async_block_escaped)]
    ReturnAsyncBlock {
        #[primary_span]
        span: Span,
    },
    #[label(borrowck_returned_ref_escaped)]
    ReturnRef {
        #[primary_span]
        span: Span,
    },
}

#[derive(Diagnostic)]
#[diag(borrowck_lifetime_constraints_error)]
pub(crate) struct LifetimeOutliveErr {
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
pub(crate) enum LifetimeReturnCategoryErr<'a> {
    #[label(borrowck_returned_lifetime_wrong)]
    WrongReturn {
        #[primary_span]
        span: Span,
        mir_def_name: &'a str,
        outlived_fr_name: RegionName,
        fr_name: &'a RegionName,
    },
    #[label(borrowck_returned_lifetime_short)]
    ShortReturn {
        #[primary_span]
        span: Span,
        category_desc: &'static str,
        free_region_name: &'a RegionName,
        outlived_fr_name: RegionName,
    },
}

#[derive(Subdiagnostic)]
pub(crate) enum RequireStaticErr {
    #[note(borrowck_used_impl_require_static)]
    UsedImpl {
        #[primary_span]
        multi_span: MultiSpan,
    },
}

#[derive(Subdiagnostic)]
pub(crate) enum CaptureVarPathUseCause {
    #[label(borrowck_borrow_due_to_use_generator)]
    BorrowInGenerator {
        #[primary_span]
        path_span: Span,
    },
    #[label(borrowck_use_due_to_use_generator)]
    UseInGenerator {
        #[primary_span]
        path_span: Span,
    },
    #[label(borrowck_assign_due_to_use_generator)]
    AssignInGenerator {
        #[primary_span]
        path_span: Span,
    },
    #[label(borrowck_assign_part_due_to_use_generator)]
    AssignPartInGenerator {
        #[primary_span]
        path_span: Span,
    },
    #[label(borrowck_borrow_due_to_use_closure)]
    BorrowInClosure {
        #[primary_span]
        path_span: Span,
    },
    #[label(borrowck_use_due_to_use_closure)]
    UseInClosure {
        #[primary_span]
        path_span: Span,
    },
    #[label(borrowck_assign_due_to_use_closure)]
    AssignInClosure {
        #[primary_span]
        path_span: Span,
    },
    #[label(borrowck_assign_part_due_to_use_closure)]
    AssignPartInClosure {
        #[primary_span]
        path_span: Span,
    },
}

#[derive(Subdiagnostic)]
pub(crate) enum CaptureVarKind {
    #[label(borrowck_capture_immute)]
    Immute {
        #[primary_span]
        kind_span: Span,
    },
    #[label(borrowck_capture_mut)]
    Mut {
        #[primary_span]
        kind_span: Span,
    },
    #[label(borrowck_capture_move)]
    Move {
        #[primary_span]
        kind_span: Span,
    },
}

#[derive(Subdiagnostic)]
pub(crate) enum CaptureVarCause {
    #[label(borrowck_var_borrow_by_use_place_in_generator)]
    BorrowUsePlaceGenerator {
        place: String,
        #[primary_span]
        var_span: Span,
    },
    #[label(borrowck_var_borrow_by_use_place_in_closure)]
    BorrowUsePlaceClosure {
        place: String,
        #[primary_span]
        var_span: Span,
    },
}

#[derive(Diagnostic)]
#[diag(borrowck_cannot_move_when_borrowed, code = "E0505")]
pub(crate) struct MoveBorrow<'a> {
    pub place: &'a str,
    pub borrow_place: &'a str,
    pub value_place: &'a str,
    #[primary_span]
    #[label(move_label)]
    pub span: Span,
    #[label]
    pub borrow_span: Span,
}

#[derive(Diagnostic)]
#[diag(borrowck_opaque_type_non_generic_param, code = "E0792")]
pub(crate) struct NonGenericOpaqueTypeParam<'a, 'tcx> {
    pub ty: GenericArg<'tcx>,
    pub kind: &'a str,
    #[primary_span]
    pub span: Span,
    #[label]
    pub param_span: Span,
}
