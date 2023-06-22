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
#[note(borrowck_cannot_escape)]
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
    Immut {
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
        is_single_var: bool,
        place: String,
        #[primary_span]
        var_span: Span,
    },
    #[label(borrowck_var_borrow_by_use_place_in_closure)]
    BorrowUsePlaceClosure {
        is_single_var: bool,
        place: String,
        #[primary_span]
        var_span: Span,
    },
    #[label(borrowck_var_borrow_by_use_in_generator)]
    BorrowUseInGenerator {
        #[primary_span]
        var_span: Span,
    },
    #[label(borrowck_var_borrow_by_use_in_closure)]
    BorrowUseInClosure {
        #[primary_span]
        var_span: Span,
    },
    #[label(borrowck_var_move_by_use_in_generator)]
    MoveUseInGenerator {
        #[primary_span]
        var_span: Span,
    },
    #[label(borrowck_var_move_by_use_in_closure)]
    MoveUseInClosure {
        #[primary_span]
        var_span: Span,
    },
    #[label(borrowck_var_first_borrow_by_use_place_in_generator)]
    FirstBorrowUsePlaceGenerator {
        place: String,
        #[primary_span]
        var_span: Span,
    },
    #[label(borrowck_var_first_borrow_by_use_place_in_closure)]
    FirstBorrowUsePlaceClosure {
        place: String,
        #[primary_span]
        var_span: Span,
    },
    #[label(borrowck_var_second_borrow_by_use_place_in_generator)]
    SecondBorrowUsePlaceGenerator {
        place: String,
        #[primary_span]
        var_span: Span,
    },
    #[label(borrowck_var_second_borrow_by_use_place_in_closure)]
    SecondBorrowUsePlaceClosure {
        place: String,
        #[primary_span]
        var_span: Span,
    },
    #[label(borrowck_var_mutable_borrow_by_use_place_in_closure)]
    MutableBorrowUsePlaceClosure {
        place: String,
        #[primary_span]
        var_span: Span,
    },
    #[label(borrowck_partial_var_move_by_use_in_generator)]
    PartialMoveUseInGenerator {
        #[primary_span]
        var_span: Span,
        is_partial: bool,
    },
    #[label(borrowck_partial_var_move_by_use_in_closure)]
    PartialMoveUseInClosure {
        #[primary_span]
        var_span: Span,
        is_partial: bool,
    },
}

#[derive(Diagnostic)]
#[diag(borrowck_cannot_move_when_borrowed, code = "E0505")]
pub(crate) struct MoveBorrow<'a> {
    pub place: &'a str,
    pub borrow_place: &'a str,
    pub value_place: &'a str,
    #[primary_span]
    #[label(borrowck_move_label)]
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

#[derive(Subdiagnostic)]
pub(crate) enum CaptureReasonLabel<'a> {
    #[label(borrowck_moved_due_to_call)]
    Call {
        #[primary_span]
        fn_call_span: Span,
        place_name: &'a str,
        is_partial: bool,
        is_loop_message: bool,
    },
    #[label(borrowck_moved_due_to_usage_in_operator)]
    OperatorUse {
        #[primary_span]
        fn_call_span: Span,
        place_name: &'a str,
        is_partial: bool,
        is_loop_message: bool,
    },
    #[label(borrowck_moved_due_to_implicit_into_iter_call)]
    ImplicitCall {
        #[primary_span]
        fn_call_span: Span,
        place_name: &'a str,
        is_partial: bool,
        is_loop_message: bool,
    },
    #[label(borrowck_moved_due_to_method_call)]
    MethodCall {
        #[primary_span]
        fn_call_span: Span,
        place_name: &'a str,
        is_partial: bool,
        is_loop_message: bool,
    },
    #[label(borrowck_moved_due_to_await)]
    Await {
        #[primary_span]
        fn_call_span: Span,
        place_name: &'a str,
        is_partial: bool,
        is_loop_message: bool,
    },
    #[label(borrowck_value_moved_here)]
    MovedHere {
        #[primary_span]
        move_span: Span,
        is_partial: bool,
        is_move_msg: bool,
        is_loop_message: bool,
    },
    #[label(borrowck_consider_borrow_type_contents)]
    BorrowContent {
        #[primary_span]
        var_span: Span,
    },
}

#[derive(Subdiagnostic)]
pub(crate) enum CaptureReasonNote {
    #[note(borrowck_moved_a_fn_once_in_call)]
    FnOnceMoveInCall {
        #[primary_span]
        var_span: Span,
    },
    #[note(borrowck_calling_operator_moves_lhs)]
    LhsMoveByOperator {
        #[primary_span]
        span: Span,
    },
    #[note(borrowck_func_take_self_moved_place)]
    FuncTakeSelf {
        func: String,
        place_name: String,
        #[primary_span]
        span: Span,
    },
}

#[derive(Subdiagnostic)]
pub(crate) enum CaptureReasonSuggest<'tcx> {
    #[suggestion(
        borrowck_suggest_iterate_over_slice,
        applicability = "maybe-incorrect",
        code = "&",
        style = "verbose"
    )]
    IterateSlice {
        ty: Ty<'tcx>,
        #[primary_span]
        span: Span,
    },
    #[suggestion(
        borrowck_suggest_create_freash_reborrow,
        applicability = "maybe-incorrect",
        code = ".as_mut()",
        style = "verbose"
    )]
    FreshReborrow {
        #[primary_span]
        span: Span,
    },
}

#[derive(Subdiagnostic)]
pub(crate) enum CaptureArgLabel {
    #[label(borrowck_value_capture_here)]
    Capture {
        is_within: bool,
        #[primary_span]
        args_span: Span,
    },
    #[label(borrowck_move_out_place_here)]
    MoveOutPlace {
        place: String,
        #[primary_span]
        args_span: Span,
    },
}

#[derive(Subdiagnostic)]
pub(crate) enum OnClosureNote<'a> {
    #[note(borrowck_closure_invoked_twice)]
    InvokedTwice {
        place_name: &'a str,
        #[primary_span]
        span: Span,
    },
    #[note(borrowck_closure_moved_twice)]
    MovedTwice {
        place_name: &'a str,
        #[primary_span]
        span: Span,
    },
}

#[derive(Subdiagnostic)]
pub(crate) enum TypeNoCopy<'a, 'tcx> {
    #[label(borrowck_ty_no_impl_copy)]
    Label {
        is_partial_move: bool,
        ty: Ty<'tcx>,
        place: &'a str,
        #[primary_span]
        span: Span,
    },
    #[note(borrowck_ty_no_impl_copy)]
    Note { is_partial_move: bool, ty: Ty<'tcx>, place: &'a str },
}
