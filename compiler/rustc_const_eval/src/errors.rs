use rustc_hir::ConstContext;
use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};
use rustc_middle::ty::Ty;
use rustc_span::Span;

#[derive(Diagnostic)]
#[diag(const_eval_dangling_ptr_in_final)]
pub(crate) struct DanglingPtrInFinal {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(const_eval_unstable_in_stable)]
pub(crate) struct UnstableInStable {
    pub gate: String,
    #[primary_span]
    pub span: Span,
    #[suggestion(
        const_eval_unstable_sugg,
        code = "#[rustc_const_unstable(feature = \"...\", issue = \"...\")]\n",
        applicability = "has-placeholders"
    )]
    #[suggestion(
        const_eval_bypass_sugg,
        code = "#[rustc_allow_const_fn_unstable({gate})]\n",
        applicability = "has-placeholders"
    )]
    pub attr_span: Span,
}

#[derive(Diagnostic)]
#[diag(const_eval_thread_local_access, code = "E0625")]
pub(crate) struct NonConstOpErr {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(const_eval_static_access, code = "E0013")]
#[help]
pub(crate) struct StaticAccessErr {
    #[primary_span]
    pub span: Span,
    pub kind: ConstContext,
    #[note(const_eval_teach_note)]
    #[help(const_eval_teach_help)]
    pub teach: Option<()>,
}

#[derive(Diagnostic)]
#[diag(const_eval_raw_ptr_to_int)]
#[note]
#[note(const_eval_note2)]
pub(crate) struct RawPtrToIntErr {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(const_eval_raw_ptr_comparison)]
#[note]
pub(crate) struct RawPtrComparisonErr {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(const_eval_panic_non_str)]
pub(crate) struct PanicNonStrErr {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(const_eval_mut_deref, code = "E0658")]
pub(crate) struct MutDerefErr {
    #[primary_span]
    pub span: Span,
    pub kind: ConstContext,
}

#[derive(Diagnostic)]
#[diag(const_eval_transient_mut_borrow, code = "E0658")]
pub(crate) struct TransientMutBorrowErr {
    #[primary_span]
    pub span: Span,
    pub kind: ConstContext,
}

#[derive(Diagnostic)]
#[diag(const_eval_transient_mut_borrow_raw, code = "E0658")]
pub(crate) struct TransientMutBorrowErrRaw {
    #[primary_span]
    pub span: Span,
    pub kind: ConstContext,
}

#[derive(Diagnostic)]
#[diag(const_eval_max_num_nodes_in_const)]
pub(crate) struct MaxNumNodesInConstErr {
    #[primary_span]
    pub span: Option<Span>,
    pub global_const_id: String,
}

#[derive(Diagnostic)]
#[diag(const_eval_unallowed_fn_pointer_call)]
pub(crate) struct UnallowedFnPointerCall {
    #[primary_span]
    pub span: Span,
    pub kind: ConstContext,
}

#[derive(Diagnostic)]
#[diag(const_eval_unstable_const_fn)]
pub(crate) struct UnstableConstFn {
    #[primary_span]
    pub span: Span,
    pub def_path: String,
}

#[derive(Diagnostic)]
#[diag(const_eval_unallowed_mutable_refs, code = "E0764")]
pub(crate) struct UnallowedMutableRefs {
    #[primary_span]
    pub span: Span,
    pub kind: ConstContext,
    #[note(const_eval_teach_note)]
    pub teach: Option<()>,
}

#[derive(Diagnostic)]
#[diag(const_eval_unallowed_mutable_refs_raw, code = "E0764")]
pub(crate) struct UnallowedMutableRefsRaw {
    #[primary_span]
    pub span: Span,
    pub kind: ConstContext,
    #[note(const_eval_teach_note)]
    pub teach: Option<()>,
}
#[derive(Diagnostic)]
#[diag(const_eval_non_const_fmt_macro_call, code = "E0015")]
pub(crate) struct NonConstFmtMacroCall {
    #[primary_span]
    pub span: Span,
    pub kind: ConstContext,
}

#[derive(Diagnostic)]
#[diag(const_eval_non_const_fn_call, code = "E0015")]
pub(crate) struct NonConstFnCall {
    #[primary_span]
    pub span: Span,
    pub def_path_str: String,
    pub kind: ConstContext,
}

#[derive(Diagnostic)]
#[diag(const_eval_unallowed_op_in_const_context)]
pub(crate) struct UnallowedOpInConstContext {
    #[primary_span]
    pub span: Span,
    pub msg: String,
}

#[derive(Diagnostic)]
#[diag(const_eval_unallowed_heap_allocations, code = "E0010")]
pub(crate) struct UnallowedHeapAllocations {
    #[primary_span]
    #[label]
    pub span: Span,
    pub kind: ConstContext,
    #[note(const_eval_teach_note)]
    pub teach: Option<()>,
}

#[derive(Diagnostic)]
#[diag(const_eval_unallowed_inline_asm, code = "E0015")]
pub(crate) struct UnallowedInlineAsm {
    #[primary_span]
    pub span: Span,
    pub kind: ConstContext,
}

#[derive(Diagnostic)]
#[diag(const_eval_unsupported_untyped_pointer)]
#[note]
pub(crate) struct UnsupportedUntypedPointer {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(const_eval_interior_mutable_data_refer, code = "E0492")]
pub(crate) struct InteriorMutableDataRefer {
    #[primary_span]
    #[label]
    pub span: Span,
    #[help]
    pub opt_help: Option<()>,
    pub kind: ConstContext,
    #[note(const_eval_teach_note)]
    pub teach: Option<()>,
}

#[derive(Diagnostic)]
#[diag(const_eval_interior_mutability_borrow)]
pub(crate) struct InteriorMutabilityBorrow {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(const_eval_erroneous_constant)]
pub(crate) struct ErroneousConstUsed {
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
#[note(const_eval_non_const_impl)]
pub(crate) struct NonConstImplNote {
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic, PartialEq, Eq, Clone)]
#[note(const_eval_frame_note)]
pub struct FrameNote {
    #[primary_span]
    pub span: Span,
    pub times: i32,
    pub where_: &'static str,
    pub instance: String,
}

#[derive(Subdiagnostic)]
#[note(const_eval_raw_bytes)]
pub struct RawBytesNote {
    pub size: u64,
    pub align: u64,
    pub bytes: String,
}

#[derive(Diagnostic)]
#[diag(const_eval_for_loop_into_iter_non_const, code = "E0015")]
pub struct NonConstForLoopIntoIter<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub kind: ConstContext,
}

#[derive(Diagnostic)]
#[diag(const_eval_question_branch_non_const, code = "E0015")]
pub struct NonConstQuestionBranch<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub kind: ConstContext,
}

#[derive(Diagnostic)]
#[diag(const_eval_question_from_residual_non_const, code = "E0015")]
pub struct NonConstQuestionFromResidual<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub kind: ConstContext,
}

#[derive(Diagnostic)]
#[diag(const_eval_try_block_from_output_non_const, code = "E0015")]
pub struct NonConstTryBlockFromOutput<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub kind: ConstContext,
}

#[derive(Diagnostic)]
#[diag(const_eval_await_non_const, code = "E0015")]
pub struct NonConstAwait<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub kind: ConstContext,
}

#[derive(Diagnostic)]
#[diag(const_eval_closure_non_const, code = "E0015")]
pub struct NonConstClosure {
    #[primary_span]
    pub span: Span,
    pub kind: ConstContext,
    #[subdiagnostic]
    pub note: Option<NonConstClosureNote>,
}

#[derive(Subdiagnostic)]
pub enum NonConstClosureNote {
    #[note(const_eval_closure_fndef_not_const)]
    FnDef {
        #[primary_span]
        span: Span,
    },
    #[note(const_eval_fn_ptr_call)]
    FnPtr,
    #[note(const_eval_closure_call)]
    Closure,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(const_eval_consider_dereferencing, applicability = "machine-applicable")]
pub struct ConsiderDereferencing {
    pub deref: String,
    #[suggestion_part(code = "{deref}")]
    pub span: Span,
    #[suggestion_part(code = "{deref}")]
    pub rhs_span: Span,
}

#[derive(Diagnostic)]
#[diag(const_eval_operator_non_const, code = "E0015")]
pub struct NonConstOperator {
    #[primary_span]
    pub span: Span,
    pub kind: ConstContext,
    #[subdiagnostic]
    pub sugg: Option<ConsiderDereferencing>,
}

#[derive(Diagnostic)]
#[diag(const_eval_deref_coercion_non_const, code = "E0015")]
#[note]
pub struct NonConstDerefCoercion<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub kind: ConstContext,
    pub target_ty: Ty<'tcx>,
    #[note(const_eval_target_note)]
    pub deref_target: Option<Span>,
}

#[derive(Diagnostic)]
#[diag(const_eval_live_drop, code = "E0493")]
pub struct LiveDrop<'tcx> {
    #[primary_span]
    #[label]
    pub span: Span,
    pub kind: ConstContext,
    pub dropped_ty: Ty<'tcx>,
    #[label(const_eval_dropped_at_label)]
    pub dropped_at: Option<Span>,
}

#[derive(LintDiagnostic)]
#[diag(const_eval_align_check_failed)]
pub struct AlignmentCheckFailed {
    pub has: u64,
    pub required: u64,
    #[subdiagnostic]
    pub frames: Vec<FrameNote>,
}

#[derive(Diagnostic)]
#[diag(const_eval_error, code = "E0080")]
pub struct ConstEvalError {
    #[primary_span]
    pub span: Span,
    /// One of "const", "const_with_path", and "static"
    pub error_kind: &'static str,
    pub instance: String,
    #[subdiagnostic]
    pub frame_notes: Vec<FrameNote>,
}

#[derive(Diagnostic)]
#[diag(const_eval_nullary_intrinsic_fail)]
pub struct NullaryIntrinsicError {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(const_eval_undefined_behavior, code = "E0080")]
pub struct UndefinedBehavior {
    #[primary_span]
    pub span: Span,
    #[note(const_eval_undefined_behavior_note)]
    pub ub_note: Option<()>,
    #[subdiagnostic]
    pub frames: Vec<FrameNote>,
    #[subdiagnostic]
    pub raw_bytes: RawBytesNote,
}
