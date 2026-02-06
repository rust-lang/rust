use rustc_errors::MultiSpan;
use rustc_errors::codes::*;
use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};
use rustc_middle::ty::{GenericArg, Ty};
use rustc_span::Span;

use crate::diagnostics::RegionName;

#[derive(Diagnostic)]
#[diag("cannot move a value of type `{$ty}`", code = E0161)]
pub(crate) struct MoveUnsized<'tcx> {
    pub ty: Ty<'tcx>,
    #[primary_span]
    #[label("the size of `{$ty}` cannot be statically determined")]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("higher-ranked lifetime error")]
pub(crate) struct HigherRankedLifetimeError {
    #[subdiagnostic]
    pub cause: Option<HigherRankedErrorCause>,
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
pub(crate) enum HigherRankedErrorCause {
    #[note("could not prove `{$predicate}`")]
    CouldNotProve { predicate: String },
    #[note("could not normalize `{$value}`")]
    CouldNotNormalize { value: String },
}

#[derive(Diagnostic)]
#[diag("higher-ranked subtype error")]
pub(crate) struct HigherRankedSubtypeError {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("`{$kind}` does not live long enough")]
pub(crate) struct GenericDoesNotLiveLongEnough {
    pub kind: String,
    #[primary_span]
    pub span: Span,
}

#[derive(LintDiagnostic)]
#[diag("variable does not need to be mutable")]
pub(crate) struct VarNeedNotMut {
    #[suggestion(
        "remove this `mut`",
        style = "short",
        applicability = "machine-applicable",
        code = ""
    )]
    pub span: Span,
}
#[derive(Diagnostic)]
#[diag("captured variable cannot escape `FnMut` closure body")]
#[note("`FnMut` closures only have access to their captured variables while they are executing...")]
#[note("...therefore, they cannot allow references to captured variables to escape")]
pub(crate) struct FnMutError {
    #[primary_span]
    pub span: Span,
    #[subdiagnostic]
    pub ty_err: FnMutReturnTypeErr,
}

#[derive(Subdiagnostic)]
pub(crate) enum VarHereDenote {
    #[label("variable captured here")]
    Captured {
        #[primary_span]
        span: Span,
    },
    #[label("variable defined here")]
    Defined {
        #[primary_span]
        span: Span,
    },
    #[label("inferred to be a `FnMut` closure")]
    FnMutInferred {
        #[primary_span]
        span: Span,
    },
}

#[derive(Subdiagnostic)]
pub(crate) enum FnMutReturnTypeErr {
    #[label(
        "returns a closure that contains a reference to a captured variable, which then escapes the closure body"
    )]
    ReturnClosure {
        #[primary_span]
        span: Span,
    },
    #[label(
        "returns an `async` block that contains a reference to a captured variable, which then escapes the closure body"
    )]
    ReturnAsyncBlock {
        #[primary_span]
        span: Span,
    },
    #[label("returns a reference to a captured variable which escapes the closure body")]
    ReturnRef {
        #[primary_span]
        span: Span,
    },
}

#[derive(Diagnostic)]
#[diag("lifetime may not live long enough")]
pub(crate) struct LifetimeOutliveErr {
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
pub(crate) enum LifetimeReturnCategoryErr<'a> {
    #[label(
        "{$mir_def_name} was supposed to return data with lifetime `{$outlived_fr_name}` but it is returning data with lifetime `{$fr_name}`"
    )]
    WrongReturn {
        #[primary_span]
        span: Span,
        mir_def_name: &'a str,
        outlived_fr_name: RegionName,
        fr_name: &'a RegionName,
    },
    #[label(
        "{$category_desc}requires that `{$free_region_name}` must outlive `{$outlived_fr_name}`"
    )]
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
    #[note("the used `impl` has a `'static` requirement")]
    UsedImpl {
        #[primary_span]
        multi_span: MultiSpan,
    },
}

#[derive(Subdiagnostic)]
pub(crate) enum CaptureVarPathUseCause {
    #[label("borrow occurs due to use in coroutine")]
    BorrowInCoroutine {
        #[primary_span]
        path_span: Span,
    },
    #[label("use occurs due to use in coroutine")]
    UseInCoroutine {
        #[primary_span]
        path_span: Span,
    },
    #[label("assign occurs due to use in coroutine")]
    AssignInCoroutine {
        #[primary_span]
        path_span: Span,
    },
    #[label("assign to part occurs due to use in coroutine")]
    AssignPartInCoroutine {
        #[primary_span]
        path_span: Span,
    },
    #[label("borrow occurs due to use in closure")]
    BorrowInClosure {
        #[primary_span]
        path_span: Span,
    },
    #[label("use occurs due to use in closure")]
    UseInClosure {
        #[primary_span]
        path_span: Span,
    },
    #[label("assignment occurs due to use in closure")]
    AssignInClosure {
        #[primary_span]
        path_span: Span,
    },
    #[label("assignment to part occurs due to use in closure")]
    AssignPartInClosure {
        #[primary_span]
        path_span: Span,
    },
}

#[derive(Subdiagnostic)]
pub(crate) enum CaptureVarKind {
    #[label("capture is immutable because of use here")]
    Immut {
        #[primary_span]
        kind_span: Span,
    },
    #[label("capture is mutable because of use here")]
    Mut {
        #[primary_span]
        kind_span: Span,
    },
    #[label("capture is moved because of use here")]
    Move {
        #[primary_span]
        kind_span: Span,
    },
}

#[derive(Subdiagnostic)]
pub(crate) enum CaptureVarCause {
    #[label(
        "{$is_single_var ->
            *[true] borrow occurs
            [false] borrows occur
        } due to use of {$place} in coroutine"
    )]
    BorrowUsePlaceCoroutine {
        is_single_var: bool,
        place: String,
        #[primary_span]
        var_span: Span,
    },
    #[label(
        "{$is_single_var ->
            *[true] borrow occurs
            [false] borrows occur
        } due to use of {$place} in closure"
    )]
    BorrowUsePlaceClosure {
        is_single_var: bool,
        place: String,
        #[primary_span]
        var_span: Span,
    },
    #[label("borrow occurs due to use in coroutine")]
    BorrowUseInCoroutine {
        #[primary_span]
        var_span: Span,
    },
    #[label("borrow occurs due to use in closure")]
    BorrowUseInClosure {
        #[primary_span]
        var_span: Span,
    },
    #[label("move occurs due to use in coroutine")]
    MoveUseInCoroutine {
        #[primary_span]
        var_span: Span,
    },
    #[label("move occurs due to use in closure")]
    MoveUseInClosure {
        #[primary_span]
        var_span: Span,
    },
    #[label("first borrow occurs due to use of {$place} in coroutine")]
    FirstBorrowUsePlaceCoroutine {
        place: String,
        #[primary_span]
        var_span: Span,
    },
    #[label("first borrow occurs due to use of {$place} in closure")]
    FirstBorrowUsePlaceClosure {
        place: String,
        #[primary_span]
        var_span: Span,
    },
    #[label("second borrow occurs due to use of {$place} in coroutine")]
    SecondBorrowUsePlaceCoroutine {
        place: String,
        #[primary_span]
        var_span: Span,
    },
    #[label("second borrow occurs due to use of {$place} in closure")]
    SecondBorrowUsePlaceClosure {
        place: String,
        #[primary_span]
        var_span: Span,
    },
    #[label("mutable borrow occurs due to use of {$place} in closure")]
    MutableBorrowUsePlaceClosure {
        place: String,
        #[primary_span]
        var_span: Span,
    },
    #[label(
        "variable {$is_partial ->
            [true] partially moved
            *[false] moved
        } due to use in coroutine"
    )]
    PartialMoveUseInCoroutine {
        #[primary_span]
        var_span: Span,
        is_partial: bool,
    },
    #[label(
        "variable {$is_partial ->
            [true] partially moved
            *[false] moved
        } due to use in closure"
    )]
    PartialMoveUseInClosure {
        #[primary_span]
        var_span: Span,
        is_partial: bool,
    },
}

#[derive(Diagnostic)]
#[diag("cannot move out of {$place ->
    [value] value
    *[other] {$place}
} because it is borrowed", code = E0505)]
pub(crate) struct MoveBorrow<'a> {
    pub place: &'a str,
    pub borrow_place: &'a str,
    pub value_place: &'a str,
    #[primary_span]
    #[label(
        "move out of {$value_place ->
            [value] value
            *[other] {$value_place}
        } occurs here"
    )]
    pub span: Span,
    #[label(
        "borrow of {$borrow_place ->
            [value] value
            *[other] {$borrow_place}
        } occurs here"
    )]
    pub borrow_span: Span,
}

#[derive(Diagnostic)]
#[diag("opaque type used twice with different lifetimes")]
pub(crate) struct LifetimeMismatchOpaqueParam<'tcx> {
    pub arg: GenericArg<'tcx>,
    pub prev: GenericArg<'tcx>,
    #[primary_span]
    #[label("lifetime `{$arg}` used here")]
    #[note(
        "if all non-lifetime generic parameters are the same, but the lifetime parameters differ, it is not possible to differentiate the opaque types"
    )]
    pub span: Span,
    #[label("lifetime `{$prev}` previously used here")]
    pub prev_span: Span,
}

#[derive(Subdiagnostic)]
pub(crate) enum CaptureReasonLabel<'a> {
    #[label(
        "{$place_name} {$is_partial ->
            [true] partially moved
            *[false] moved
        } due to this {$is_loop_message ->
            [true] call, in previous iteration of loop
            *[false] call
        }"
    )]
    Call {
        #[primary_span]
        fn_call_span: Span,
        place_name: &'a str,
        is_partial: bool,
        is_loop_message: bool,
    },
    #[label(
        "{$place_name} {$is_partial ->
            [true] partially moved
            *[false] moved
        } due to usage in {$is_loop_message ->
            [true] operator, in previous iteration of loop
            *[false] operator
        }"
    )]
    OperatorUse {
        #[primary_span]
        fn_call_span: Span,
        place_name: &'a str,
        is_partial: bool,
        is_loop_message: bool,
    },
    #[label(
        "{$place_name} {$is_partial ->
            [true] partially moved
            *[false] moved
        } due to this implicit call to {$is_loop_message ->
            [true] `.into_iter()`, in previous iteration of loop
            *[false] `.into_iter()`
        }"
    )]
    ImplicitCall {
        #[primary_span]
        fn_call_span: Span,
        place_name: &'a str,
        is_partial: bool,
        is_loop_message: bool,
    },
    #[label(
        "{$place_name} {$is_partial ->
            [true] partially moved
            *[false] moved
        } due to this method {$is_loop_message ->
            [true] call, in previous iteration of loop
            *[false] call
        }"
    )]
    MethodCall {
        #[primary_span]
        fn_call_span: Span,
        place_name: &'a str,
        is_partial: bool,
        is_loop_message: bool,
    },
    #[label(
        "{$place_name} {$is_partial ->
            [true] partially moved
            *[false] moved
        } due to this {$is_loop_message ->
            [true] await, in previous iteration of loop
            *[false] await
        }"
    )]
    Await {
        #[primary_span]
        fn_call_span: Span,
        place_name: &'a str,
        is_partial: bool,
        is_loop_message: bool,
    },
    #[label(
        "value {$is_partial ->
            [true] partially moved
            *[false] moved
        } {$is_move_msg ->
            [true] into closure here
            *[false] here
        }{$is_loop_message ->
            [true] , in previous iteration of loop
            *[false] {\"\"}
        }"
    )]
    MovedHere {
        #[primary_span]
        move_span: Span,
        is_partial: bool,
        is_move_msg: bool,
        is_loop_message: bool,
    },
    #[label("help: consider calling `.as_ref()` or `.as_mut()` to borrow the type's contents")]
    BorrowContent {
        #[primary_span]
        var_span: Span,
    },
}

#[derive(Subdiagnostic)]
pub(crate) enum CaptureReasonNote {
    #[note("this value implements `FnOnce`, which causes it to be moved when called")]
    FnOnceMoveInCall {
        #[primary_span]
        var_span: Span,
    },
    #[note("calling this operator moves the value")]
    UnOpMoveByOperator {
        #[primary_span]
        span: Span,
    },
    #[note("calling this operator moves the left-hand side")]
    LhsMoveByOperator {
        #[primary_span]
        span: Span,
    },
    #[note("`{$func}` takes ownership of the receiver `self`, which moves {$place_name}")]
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
        "consider iterating over a slice of the `{$ty}`'s content to avoid moving into the `for` loop",
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
        "consider reborrowing the `Pin` instead of moving it",
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
    #[label(
        "value captured {$is_within ->
            [true] here by coroutine
            *[false] here
        }"
    )]
    Capture {
        is_within: bool,
        #[primary_span]
        args_span: Span,
    },
    #[label("{$place} is moved here")]
    MoveOutPlace {
        place: String,
        #[primary_span]
        args_span: Span,
    },
}

#[derive(Subdiagnostic)]
pub(crate) enum OnClosureNote<'a> {
    #[note(
        "closure cannot be invoked more than once because it moves the variable `{$place_name}` out of its environment"
    )]
    InvokedTwice {
        place_name: &'a str,
        #[primary_span]
        span: Span,
    },
    #[note(
        "closure cannot be moved more than once as it is not `Copy` due to moving the variable `{$place_name}` out of its environment"
    )]
    MovedTwice {
        place_name: &'a str,
        #[primary_span]
        span: Span,
    },
}

#[derive(Subdiagnostic)]
pub(crate) enum TypeNoCopy<'a, 'tcx> {
    #[label(
        "{$is_partial_move ->
            [true] partial move
            *[false] move
        } occurs because {$place} has type `{$ty}`, which does not implement the `Copy` trait"
    )]
    Label {
        is_partial_move: bool,
        ty: Ty<'tcx>,
        place: &'a str,
        #[primary_span]
        span: Span,
    },
    #[note(
        "{$is_partial_move ->
            [true] partial move
            *[false] move
        } occurs because {$place} has type `{$ty}`, which does not implement the `Copy` trait"
    )]
    Note { is_partial_move: bool, ty: Ty<'tcx>, place: &'a str },
}

#[derive(Diagnostic)]
#[diag(
    "{$arg ->
        [1] 1st
        [2] 2nd
        [3] 3rd
        *[other] {$arg}th
    } argument of `{$intrinsic}` is required to be a `const` item"
)]
pub(crate) struct SimdIntrinsicArgConst {
    #[primary_span]
    pub span: Span,
    pub arg: usize,
    pub intrinsic: String,
}

#[derive(LintDiagnostic)]
#[diag("relative drop order changing in Rust 2024")]
pub(crate) struct TailExprDropOrder {
    #[label("this temporary value will be dropped at the end of the block")]
    pub borrowed: Span,
}
