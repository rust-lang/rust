use rustc_errors::{codes::*, MultiSpan};
use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};
use rustc_middle::ty::{GenericArg, Ty};
use rustc_span::{Span, Symbol};

use crate::diagnostics::{DescribedPlace, RegionName};

#[derive(Diagnostic)]
#[diag(borrowck_move_unsized, code = E0161)]
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
        category: &'a str,
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
    #[label(borrowck_borrow_due_to_use_coroutine)]
    BorrowInCoroutine {
        #[primary_span]
        path_span: Span,
    },
    #[label(borrowck_use_due_to_use_coroutine)]
    UseInCoroutine {
        #[primary_span]
        path_span: Span,
    },
    #[label(borrowck_assign_due_to_use_coroutine)]
    AssignInCoroutine {
        #[primary_span]
        path_span: Span,
    },
    #[label(borrowck_assign_part_due_to_use_coroutine)]
    AssignPartInCoroutine {
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
    #[label(borrowck_var_borrow_by_use_place_in_coroutine)]
    BorrowUsePlaceCoroutine {
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
    #[label(borrowck_var_borrow_by_use_in_coroutine)]
    BorrowUseInCoroutine {
        #[primary_span]
        var_span: Span,
    },
    #[label(borrowck_var_borrow_by_use_in_closure)]
    BorrowUseInClosure {
        #[primary_span]
        var_span: Span,
    },
    #[label(borrowck_var_move_by_use_in_coroutine)]
    MoveUseInCoroutine {
        #[primary_span]
        var_span: Span,
    },
    #[label(borrowck_var_move_by_use_in_closure)]
    MoveUseInClosure {
        #[primary_span]
        var_span: Span,
    },
    #[label(borrowck_var_first_borrow_by_use_place_in_coroutine)]
    FirstBorrowUsePlaceCoroutine {
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
    #[label(borrowck_var_second_borrow_by_use_place_in_coroutine)]
    SecondBorrowUsePlaceCoroutine {
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
    #[label(borrowck_partial_var_move_by_use_in_coroutine)]
    PartialMoveUseInCoroutine {
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
#[diag(borrowck_cannot_move_when_borrowed, code = E0505)]
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
#[diag(borrowck_opaque_type_non_generic_param, code = E0792)]
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

#[derive(Diagnostic)]
#[diag(borrowck_simd_intrinsic_arg_const)]
pub(crate) struct SimdIntrinsicArgConst {
    #[primary_span]
    pub span: Span,
    pub arg: usize,
    pub intrinsic: String,
}

#[derive(Subdiagnostic)]
pub(crate) enum AddMoveErr {
    #[label(borrowck_data_moved_here)]
    Here {
        #[primary_span]
        binding_span: Span,
    },
    #[label(borrowck_and_data_moved_here)]
    AndHere {
        #[primary_span]
        binding_span: Span,
    },
    #[note(borrowck_moved_var_cannot_copy)]
    MovedNotCopy,
}

#[derive(Subdiagnostic)]
pub(crate) enum OnLifetimeBound<'a> {
    #[help(borrowck_consider_add_lifetime_bound)]
    Add { fr_name: &'a RegionName, outlived_fr_name: &'a RegionName },
}

#[derive(Subdiagnostic)]
pub(crate) enum FnMutBumpFn {
    #[label(borrowck_cannot_assign)]
    CannotAssign {
        #[primary_span]
        sp: Span,
    },
    #[label(borrowck_cannot_borrow_mut)]
    CannotBorrowMut {
        #[primary_span]
        sp: Span,
    },
    #[label(borrowck_expects_fnmut_not_fn)]
    AcceptFnMut {
        #[primary_span]
        span: Span,
    },
    #[label(borrowck_expects_fn_not_fnmut)]
    AcceptFn {
        #[primary_span]
        span: Span,
    },
    #[label(borrowck_empty_label)]
    EmptyLabel {
        #[primary_span]
        span: Span,
    },
    #[label(borrowck_in_this_closure)]
    Here {
        #[primary_span]
        span: Span,
    },
    #[label(borrowck_return_fnmut)]
    ReturnFnMut {
        #[primary_span]
        span: Span,
    },
}

#[derive(Subdiagnostic)]
pub(crate) enum RegionNameLabels<'a> {
    #[label(borrowck_name_this_region)]
    NameRegion {
        #[primary_span]
        span: Span,
        rg_name: &'a RegionName,
    },
    #[label(borrowck_lifetime_appears_in_type)]
    LifetimeInType {
        #[primary_span]
        span: Span,
        type_name: &'a Symbol,
        rg_name: &'a RegionName,
    },
    #[label(borrowck_lifetime_appears_in_type_of)]
    LifetimeInTypeOf {
        #[primary_span]
        span: Span,
        upvar_name: &'a Symbol,
        rg_name: &'a RegionName,
    },
    #[label(borrowck_yield_type_is_type)]
    YieldTypeIsTpye {
        #[primary_span]
        span: Span,
        type_name: &'a Symbol,
    },
    #[label(borrowck_lifetime_appears_here_in_impl)]
    LifetimeInImpl {
        #[primary_span]
        span: Span,
        rg_name: &'a RegionName,
        location: &'a str,
    },
}

#[derive(Diagnostic)]
#[diag(borrowck_type_parameter_not_used_in_trait_type_alias)]
pub(crate) struct UnusedTypeParameter<'tcx> {
    pub ty: Ty<'tcx>,
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
pub(crate) enum DefiningTypeNote<'a> {
    #[note(borrowck_define_type_with_closure_args)]
    Closure { type_name: &'a str, subsets: &'a str },
    #[note(borrowck_define_type_with_generator_args)]
    Generator { type_name: &'a str, subsets: &'a str },
    #[note(borrowck_define_type)]
    FnDef { type_name: &'a str },
    #[note(borrowck_define_const_type)]
    Const { type_name: &'a str },
    #[note(borrowck_define_inline_constant_type)]
    InlineConst { type_name: &'a str },
}

#[derive(Diagnostic)]
#[diag(borrowck_borrowed_temporary_value_dropped, code = E0716)]
pub(crate) struct TemporaryDroppedErr {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(borrowck_thread_local_outlive_function, code = E0712)]
pub(crate) struct ThreadLocalOutliveErr {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(borrowck_closure_borrowing_outlive_function, code = E0373)]
pub(crate) struct ClosureVarOutliveErr<'a> {
    pub closure_kind: &'a str,
    pub borrowed_path: &'a str,
    #[primary_span]
    #[label]
    pub closure_span: Span,
    #[label(borrowck_path_label)]
    pub capture_span: Span,
}

#[derive(Diagnostic)]
#[diag(borrowck_cannot_return_ref_to_local, code = E0515)]
pub(crate) struct ReturnRefLocalErr<'a> {
    pub return_kind: &'a str,
    pub reference: &'a str,
    pub local: &'a str,
    #[primary_span]
    #[label]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(borrowck_path_does_not_live_long_enough, code = E0597)]
pub(crate) struct PathShortLive<'a> {
    pub path: &'a str,
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(borrowck_cannot_borrow_across_destructor, code = E0713)]
pub(crate) struct BorrowAcrossDestructor {
    #[primary_span]
    pub borrow_span: Span,
}

#[derive(Diagnostic)]
#[diag(borrowck_cannot_borrow_across_coroutine_yield, code = E0626)]
pub(crate) struct BorrowAcrossCoroutineYield {
    pub coroutine_kind: String,
    #[primary_span]
    pub span: Span,
    #[label]
    pub yield_span: Span,
}

#[derive(Diagnostic)]
#[diag(borrowck_cannot_move_out_of_interior_of_drop, code = E0509)]
pub(crate) struct InteriorDropMoveErr<'a> {
    pub container_ty: Ty<'a>,
    #[primary_span]
    #[label]
    pub move_from_span: Span,
}

#[derive(Diagnostic)]
#[diag(borrowck_cannot_assign_to_borrowed, code = E0506)]
pub(crate) struct AssignBorrowErr<'a> {
    pub desc: &'a str,
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(borrowck_borrow_here_label)]
    pub borrow_span: Span,
}

#[derive(Diagnostic)]
#[diag(borrowck_cannot_uniquely_borrow_by_two_closures, code = E0524)]
pub(crate) struct TwoClosuresUniquelyBorrowErr<'a> {
    pub desc: &'a str,
    #[primary_span]
    pub new_loan_span: Span,
    #[label]
    pub old_load_end_span: Option<Span>,
    #[label(borrowck_new_span_label)]
    pub diff_span: Option<Span>,
    #[subdiagnostic]
    pub case: ClosureConstructLabel,
}

#[derive(Subdiagnostic)]
pub(crate) enum ClosureConstructLabel {
    #[label(borrowck_first_closure_constructed_here)]
    First {
        #[primary_span]
        old_loan_span: Span,
    },
    #[label(borrowck_closures_constructed_here)]
    Both {
        #[primary_span]
        old_loan_span: Span,
    },
}

#[derive(Diagnostic)]
#[diag(borrowck_cannot_use_when_mutably_borrowed, code = E0503)]
pub(crate) struct UseMutBorrowErr<'a> {
    pub desc: &'a str,
    pub borrow_desc: &'a str,
    #[primary_span]
    #[label]
    pub span: Span,
    #[label(borrowck_borrow_span_label)]
    pub borrow_span: Span,
}

#[derive(Diagnostic)]
pub(crate) enum MutBorrowMulti<'a> {
    #[diag(borrowck_cannot_mutably_borrow_multiply_same_span, code = E0499)]
    SameSpan {
        new_place_name: &'a str,
        place: &'a str,
        old_place: &'a str,
        is_place_empty: bool,
        #[primary_span]
        new_loan_span: Span,
        #[label]
        old_load_end_span: Option<Span>,
        #[subdiagnostic]
        eager_label: MutMultiLoopLabel<'a>,
    },
    #[diag(borrowck_cannot_mutably_borrow_multiply, code = E0499)]
    ChangedSpan {
        new_place_name: &'a str,
        place: &'a str,
        old_place: &'a str,
        is_place_empty: bool,
        is_old_place_empty: bool,
        #[primary_span]
        #[label(borrowck_second_mut_borrow_label)]
        new_loan_span: Span,
        #[label]
        old_loan_span: Span,
        #[label(borrowck_first_mut_end_label)]
        old_load_end_span: Option<Span>,
    },
}

#[derive(Subdiagnostic)]
#[label(borrowck_mutably_borrow_multiply_loop_label)]
pub(crate) struct MutMultiLoopLabel<'a> {
    pub new_place_name: &'a str,
    pub place: &'a str,
    pub is_place_empty: bool,
    #[primary_span]
    pub new_loan_span: Span,
}

#[derive(Diagnostic)]
#[diag(borrowck_cannot_uniquely_borrow_by_one_closure, code = E0500)]
pub(crate) struct ClosureUniquelyBorrowErr<'a> {
    #[primary_span]
    #[label]
    pub new_loan_span: Span,
    pub container_name: &'a str,
    pub desc_new: &'a str,
    pub opt_via: &'a str,
    #[label(borrowck_occurs_label)]
    pub old_loan_span: Span,
    pub noun_old: &'a str,
    pub old_opt_via: &'a str,
    #[label(borrowck_ends_label)]
    pub previous_end_span: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(borrowck_cannot_reborrow_already_uniquely_borrowed, code = E0501)]
pub(crate) struct ClosureReBorrowErr<'a> {
    #[primary_span]
    #[label]
    pub new_loan_span: Span,
    pub container_name: &'a str,
    pub desc_new: &'a str,
    pub opt_via: &'a str,
    pub kind_new: &'a str,
    #[label(borrowck_occurs_label)]
    pub old_loan_span: Span,
    pub old_opt_via: &'a str,
    #[label(borrowck_ends_label)]
    pub previous_end_span: Option<Span>,
    pub second_borrow_desc: &'a str,
}

#[derive(Subdiagnostic)]
pub(crate) enum BorrowOccurLabel<'a> {
    #[label(borrowck_borrow_occurs_here)]
    Here {
        #[primary_span]
        span: Span,
        kind: &'a str,
    },
    #[label(borrowck_borrow_occurs_here_overlap)]
    HereOverlap {
        #[primary_span]
        span: Span,
        kind_new: &'a str,
        msg_new: &'a str,
        msg_old: &'a str,
    },
    #[label(borrowck_borrow_occurs_here_via)]
    HereVia {
        #[primary_span]
        span: Span,
        kind_old: &'a str,
        is_msg_old_empty: bool,
        msg_old: &'a str,
    },
}

#[derive(Diagnostic)]
#[diag(borrowck_cannot_reborrow_already_borrowed, code = E0502)]
pub(crate) struct ReborrowBorrowedErr<'a> {
    pub desc_new: &'a str,
    pub is_msg_new_empty: bool,
    pub msg_new: &'a str,
    pub kind_new: &'a str,
    pub noun_old: &'a str,
    pub kind_old: &'a str,
    pub is_msg_old_empty: bool,
    pub msg_old: &'a str,
    #[primary_span]
    pub span: Span,
    #[label]
    pub old_load_end_span: Option<Span>,
    #[subdiagnostic]
    pub new_occur: BorrowOccurLabel<'a>,
    #[subdiagnostic]
    pub old_occur: BorrowOccurLabel<'a>,
}

#[derive(Diagnostic)]
pub(crate) enum ReassignImmut<'a> {
    #[diag(borrowck_cannot_reassign_immutable_arg, code = E0384)]
    Arg {
        #[primary_span]
        span: Span,
        place: &'a str,
    },
    #[diag(borrowck_cannot_reassign_immutable_var, code = E0384)]
    Var {
        #[primary_span]
        span: Span,
        place: &'a str,
    },
}

#[derive(Subdiagnostic)]
#[help(borrowck_modify_ty_methods_help)]
pub(crate) struct OnModifyTy<'tcx> {
    pub ty: Ty<'tcx>,
}

#[derive(Subdiagnostic)]
pub(crate) enum BorrowMutUpvarLable<'a> {
    #[label(borrowck_upvar_need_mut_due_to_borrow)]
    MutBorrow {
        #[primary_span]
        span: Span,
        upvar: &'a str,
        place: String,
    },
    #[label(borrowck_upvar_need_mut_due_to_mutation)]
    Mutation {
        #[primary_span]
        span: Span,
        upvar: &'a str,
        place: String,
    },
}

#[derive(Diagnostic)]
pub(crate) enum MutBorrowErr {
    #[diag(borrowck_mut_borrow_place_declared_immute, code = E0596)]
    PlaceDeclaredImmute {
        #[primary_span]
        span: Span,
        place: DescribedPlace,
    },
    #[diag(borrowck_mut_borrow_symbol_declared_immute, code = E0596)]
    SymbolDeclaredImmute {
        #[primary_span]
        span: Span,
        place: DescribedPlace,
        name: String,
    },
    #[diag(borrowck_mut_borrow_place_in_pattern_guard_immute, code = E0596)]
    PatternGuardImmute {
        #[primary_span]
        span: Span,
        place: DescribedPlace,
    },
    #[diag(borrowck_mut_borrow_symbol_static, code = E0596)]
    SymbolStatic {
        #[primary_span]
        span: Span,
        place: DescribedPlace,
        static_name: String,
    },
    #[diag(borrowck_mut_borrow_place_static, code = E0596)]
    PlaceStatic {
        #[primary_span]
        span: Span,
        place: DescribedPlace,
    },
    #[diag(borrowck_mut_borrow_self_in_fn, code = E0596)]
    CapturedInFn {
        #[primary_span]
        span: Span,
        place: DescribedPlace,
    },
    #[diag(borrowck_mut_borrow_upvar_in_fn, code = E0596)]
    UpvarInFn {
        #[primary_span]
        span: Span,
        place: DescribedPlace,
    },
    #[diag(borrowck_mut_borrow_self_behind_const_pointer, code = E0596)]
    SelfBehindRawPointer {
        #[primary_span]
        span: Span,
        place: DescribedPlace,
    },
    #[diag(borrowck_mut_borrow_self_behind_ref, code = E0596)]
    SelfBehindSharedRef {
        #[primary_span]
        span: Span,
        place: DescribedPlace,
    },
    #[diag(borrowck_mut_borrow_self_behind_deref, code = E0596)]
    SelfBehindDeref {
        #[primary_span]
        span: Span,
        place: DescribedPlace,
        name: String,
    },
    #[diag(borrowck_mut_borrow_self_behind_index, code = E0596)]
    SelfBehindIndex {
        #[primary_span]
        span: Span,
        place: DescribedPlace,
        name: String,
    },
    #[diag(borrowck_mut_borrow_data_behind_const_pointer, code = E0596)]
    DataBehindRawPointer {
        #[primary_span]
        span: Span,
    },
    #[diag(borrowck_mut_borrow_data_behind_ref, code = E0596)]
    DataBehindSharedRef {
        #[primary_span]
        span: Span,
    },
    #[diag(borrowck_mut_borrow_data_behind_deref, code = E0596)]
    DataBehindDeref {
        #[primary_span]
        span: Span,
        name: String,
    },
    #[diag(borrowck_mut_borrow_data_behind_index, code = E0596)]
    DataBehindIndex {
        #[primary_span]
        span: Span,
        name: String,
    },
}

#[derive(Diagnostic)]
pub(crate) enum AssignErr {
    #[diag(borrowck_assign_place_declared_immute, code = E0594)]
    PlaceDeclaredImmute {
        #[primary_span]
        span: Span,
        place: DescribedPlace,
    },
    #[diag(borrowck_assign_symbol_declared_immute, code = E0594)]
    SymbolDeclaredImmute {
        #[primary_span]
        span: Span,
        place: DescribedPlace,
        name: String,
    },
    #[diag(borrowck_assign_place_in_pattern_guard_immute, code = E0594)]
    PatternGuardImmute {
        #[primary_span]
        span: Span,
        place: DescribedPlace,
    },
    #[diag(borrowck_assign_symbol_static, code = E0594)]
    SymbolStatic {
        #[primary_span]
        span: Span,
        place: DescribedPlace,
        static_name: String,
    },
    #[diag(borrowck_assign_place_static, code = E0594)]
    PlaceStatic {
        #[primary_span]
        span: Span,
        place: DescribedPlace,
    },
    #[diag(borrowck_assign_place_in_fn, code = E0594)]
    CapturedInFn {
        #[primary_span]
        span: Span,
        place: DescribedPlace,
    },
    #[diag(borrowck_assign_upvar_in_fn, code = E0594)]
    UpvarInFn {
        #[primary_span]
        span: Span,
        place: DescribedPlace,
    },
    #[diag(borrowck_assign_place_behind_const_pointer, code = E0594)]
    PlaceBehindRawPointer {
        #[primary_span]
        span: Span,
        place: DescribedPlace,
    },
    #[diag(borrowck_assign_place_behind_ref, code = E0594)]
    PlaceBehindSharedRef {
        #[primary_span]
        span: Span,
        place: DescribedPlace,
    },
    #[diag(borrowck_assign_place_behind_deref, code = E0594)]
    PlaceBehindDeref {
        #[primary_span]
        span: Span,
        place: DescribedPlace,
        name: String,
    },
    #[diag(borrowck_assign_place_behind_index, code = E0594)]
    PlaceBehindIndex {
        #[primary_span]
        span: Span,
        place: DescribedPlace,
        name: String,
    },
    #[diag(borrowck_assign_data_behind_const_pointer, code = E0594)]
    DataBehindRawPointer {
        #[primary_span]
        span: Span,
    },
    #[diag(borrowck_assign_data_behind_ref, code = E0594)]
    DataBehindSharedRef {
        #[primary_span]
        span: Span,
    },
    #[diag(borrowck_assign_data_behind_deref, code = E0594)]
    DataBehindDeref {
        #[primary_span]
        span: Span,
        name: String,
    },
    #[diag(borrowck_assign_data_behind_index, code = E0594)]
    DataBehindIndex {
        #[primary_span]
        span: Span,
        name: String,
    },
}
