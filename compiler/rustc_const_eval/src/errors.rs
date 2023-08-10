use rustc_errors::{
    AddToDiagnostic, Diagnostic, DiagnosticArgValue, DiagnosticBuilder, DiagnosticMessage,
    EmissionGuarantee, Handler, IntoDiagnostic, IntoDiagnosticArg, SubdiagnosticMessage,
};
use rustc_hir::ConstContext;
use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};
use rustc_middle::mir::interpret::{
    CheckInAllocMsg, ExpectedKind, InterpError, InvalidMetaKind, InvalidProgramInfo,
    MachineStopType, PointerKind, ResourceExhaustionInfo, UndefinedBehaviorInfo, UnsupportedOpInfo,
    ValidationErrorInfo,
};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::{ErrorGuaranteed, Span};
use rustc_target::abi::call::AdjustForForeignAbiError;
use rustc_target::abi::{Size, WrappingRange};

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

#[derive(LintDiagnostic)]
#[diag(const_eval_long_running)]
#[note]
pub struct LongRunning {
    #[help]
    pub item_span: Span,
}

#[derive(Diagnostic)]
#[diag(const_eval_long_running)]
pub struct LongRunningWarn {
    #[primary_span]
    #[label]
    pub span: Span,
    #[help]
    pub item_span: Span,
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

// FIXME(fee1-dead) do not use stringly typed `ConstContext`

#[derive(Diagnostic)]
#[diag(const_eval_match_eq_non_const, code = "E0015")]
#[note]
pub struct NonConstMatchEq<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub kind: ConstContext,
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

/// args: eagerly_translate_with_args!(
///     handler: &Handler,
///     msg: DiagnosticMessage,
///     key: str,
///     value: impl IntoDiagnosticArg,
/// )
macro_rules! eagerly_translate_with_args {
    ($handler:expr,$msg:expr$(,$key:expr,$value:expr)*$(,)?) => {
        $handler.eagerly_translate_to_string(
            $msg,
            [
                $(($key.into(), $value.into_diagnostic_arg()),)*
            ].iter()
                .map(|(a, b)| (a, b))
        )
    };
}

fn bad_pointer_message(msg: CheckInAllocMsg, handler: &Handler) -> String {
    use crate::fluent_generated::*;

    let msg = match msg {
        CheckInAllocMsg::DerefTest => const_eval_deref_test,
        CheckInAllocMsg::MemoryAccessTest => const_eval_memory_access_test,
        CheckInAllocMsg::PointerArithmeticTest => const_eval_pointer_arithmetic_test,
        CheckInAllocMsg::OffsetFromTest => const_eval_offset_from_test,
        CheckInAllocMsg::InboundsTest => const_eval_in_bounds_test,
    };

    handler.eagerly_translate_to_string(msg, [].into_iter())
}

pub struct UndefinedBehaviorInfoExt<'a>(UndefinedBehaviorInfo<'a>);

pub struct UndefinedBehaviorInfoExt2<'tcx> {
    err: UndefinedBehaviorInfo<'tcx>,
    span: Span,
    tcx: TyCtxt<'tcx>,
}

impl AddToDiagnostic for UndefinedBehaviorInfoExt2<'_> {
    fn add_to_diagnostic_with<F>(self, diag: &mut Diagnostic, _: F)
    where
        F: Fn(&mut Diagnostic, SubdiagnosticMessage) -> SubdiagnosticMessage,
    {
        use crate::fluent_generated::*;
        use UndefinedBehaviorInfo::*;

        let handler = &self.tcx.sess.parse_sess.span_diagnostic;

        // Allow untranslatable_diagnostic lint, since we
        // eagerly translate our subdiagnostics by using the
        // eagerly_translate_with_args!() macro.
        #[allow(rustc::untranslatable_diagnostic)]
        match self.err {
            #[allow(rustc::untranslatable_diagnostic)]
            Ub(str) => {
                diag.span_label(self.span, str.clone());
            }
            Unreachable => {
                diag.span_label(self.span, const_eval_unreachable);
            }
            BoundsCheckFailed { len, index } => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_bounds_check_failed,
                    "len",
                    len,
                    "index",
                    index,
                );
                diag.span_label(self.span, msg);
            }
            DivisionByZero => {
                diag.span_label(self.span, const_eval_division_by_zero);
            }
            RemainderByZero => {
                diag.span_label(self.span, const_eval_remainder_by_zero);
            }
            DivisionOverflow => {
                diag.span_label(self.span, const_eval_division_overflow);
            }
            RemainderOverflow => {
                diag.span_label(self.span, const_eval_remainder_overflow);
            }
            PointerArithOverflow => {
                diag.span_label(self.span, const_eval_pointer_arithmetic_overflow);
            }
            InvalidMeta(InvalidMetaKind::SliceTooBig) => {
                diag.span_label(self.span, const_eval_invalid_meta_slice);
            }
            InvalidMeta(InvalidMetaKind::TooBig) => {
                diag.span_label(self.span, const_eval_invalid_meta);
            }
            UnterminatedCString(ptr) => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_unterminated_c_string,
                    "pointer",
                    ptr,
                );
                diag.span_label(self.span, msg);
            }
            PointerUseAfterFree(alloc_id, msg) => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_pointer_use_after_free,
                    "alloc_id",
                    alloc_id,
                    "bad_pointer_message",
                    bad_pointer_message(msg, handler)
                );
                diag.span_label(self.span, msg);
            }
            PointerOutOfBounds { alloc_id, alloc_size, ptr_offset, ptr_size, msg } => {
                let diagnostic_message = if ptr_size == Size::ZERO {
                    const_eval_zst_pointer_out_of_bounds
                } else {
                    const_eval_pointer_out_of_bounds
                };

                let translated_msg = eagerly_translate_with_args!(
                    handler,
                    diagnostic_message,
                    "alloc_id",
                    alloc_id,
                    "alloc_size",
                    alloc_size.bytes(),
                    "ptr_offset",
                    ptr_offset,
                    "ptr_size",
                    ptr_size.bytes(),
                    "bad_pointer_message",
                    bad_pointer_message(msg, &self.tcx.sess.parse_sess.span_diagnostic),
                );
                diag.span_label(self.span, translated_msg);
            }
            DanglingIntPointer(ptr, msg) => {
                let diagnostic_message = if ptr == 0 {
                    const_eval_dangling_null_pointer
                } else {
                    const_eval_dangling_int_pointer
                };

                let mut args = vec![(
                    "bad_pointer_message".into(),
                    bad_pointer_message(msg, &self.tcx.sess.parse_sess.span_diagnostic)
                        .into_diagnostic_arg(),
                )];

                if ptr != 0 {
                    args.push((
                        "pointer".into(),
                        format!("{ptr:#x}[noalloc]").into_diagnostic_arg(),
                    ));
                }

                let msg = handler.eagerly_translate_to_string(
                    diagnostic_message,
                    args.iter().map(|(a, b)| (a, b)),
                );
                diag.span_label(self.span, msg);
            }
            AlignmentCheckFailed { required, has } => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_alignment_check_failed,
                    "required",
                    required.bytes(),
                    "has",
                    has.bytes(),
                );
                diag.span_label(self.span, msg);
            }
            WriteToReadOnly(alloc) => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_write_to_read_only,
                    "allocation",
                    alloc,
                );
                diag.span_label(self.span, msg);
            }
            DerefFunctionPointer(alloc) => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_deref_function_pointer,
                    "allocation",
                    alloc,
                );
                diag.span_label(self.span, msg);
            }
            DerefVTablePointer(alloc) => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_deref_vtable_pointer,
                    "allocation",
                    alloc,
                );
                diag.span_label(self.span, msg);
            }
            InvalidBool(b) => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_invalid_bool,
                    "value",
                    format!("{b:02x}"),
                );
                diag.span_label(self.span, msg);
            }
            InvalidChar(c) => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_invalid_char,
                    "value",
                    format!("{c:08x}"),
                );
                diag.span_label(self.span, msg);
            }
            InvalidTag(tag) => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_invalid_tag,
                    "value",
                    format!("{tag:x}"),
                );
                diag.span_label(self.span, msg);
            }
            InvalidFunctionPointer(ptr) => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_invalid_function_pointer,
                    "pointer",
                    ptr,
                );
                diag.span_label(self.span, msg);
            }
            InvalidVTablePointer(ptr) => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_invalid_vtable_pointer,
                    "pointer",
                    ptr,
                );
                diag.span_label(self.span, msg);
            }
            InvalidStr(err) => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_invalid_str,
                    "err",
                    format!("{err}"),
                );
                diag.span_label(self.span, msg);
            }
            InvalidUninitBytes(None) => {
                diag.span_label(self.span, const_eval_invalid_uninit_bytes_unknown);
            }
            InvalidUninitBytes(Some((alloc, info))) => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_invalid_uninit_bytes,
                    "alloc",
                    alloc,
                    "access",
                    info.access,
                    "uninit",
                    info.bad,
                );
                diag.span_label(self.span, msg);
            }
            DeadLocal => {
                diag.span_label(self.span, const_eval_dead_local);
            }
            ScalarSizeMismatch(info) => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_scalar_size_mismatch,
                    "target_size",
                    info.target_size,
                    "data_size",
                    info.data_size,
                );
                diag.span_label(self.span, msg);
            }
            UninhabitedEnumVariantWritten(_) => {
                diag.span_label(self.span, const_eval_uninhabited_enum_variant_written);
            }
            UninhabitedEnumVariantRead(_) => {
                diag.span_label(self.span, const_eval_uninhabited_enum_variant_read);
            }
            ValidationError(err) => {
                ValidationErrorInfoExt2 { err, tcx: self.tcx, span: self.span }
                    .add_to_diagnostic(diag);
            }
            Custom(x) => {
                let mut args = Vec::new();

                (x.add_args)(&mut |name, value| {
                    args.push((name, value));
                });

                let msg = handler.eagerly_translate_to_string(
                    (x.msg.clone())(),
                    args.iter().map(|(a, b)| (a, b)),
                );

                diag.span_label(self.span, msg);
            }
        }
    }
}

impl IntoDiagnostic<'_> for UndefinedBehaviorInfoExt<'_> {
    fn into_diagnostic(self, handler: &'_ Handler) -> DiagnosticBuilder<'_, ErrorGuaranteed> {
        use crate::fluent_generated::*;
        use UndefinedBehaviorInfo::*;
        match self.0 {
            #[allow(rustc::untranslatable_diagnostic)]
            Ub(str) => handler.struct_diagnostic(str.clone()),
            Unreachable => handler.struct_diagnostic(const_eval_unreachable),
            BoundsCheckFailed { len, index } => {
                let mut builder = handler.struct_diagnostic(const_eval_bounds_check_failed);

                builder.set_arg("len", len);
                builder.set_arg("index", index);

                builder
            }
            DivisionByZero => handler.struct_diagnostic(const_eval_division_by_zero),
            RemainderByZero => handler.struct_diagnostic(const_eval_remainder_by_zero),
            DivisionOverflow => handler.struct_diagnostic(const_eval_division_overflow),
            RemainderOverflow => handler.struct_diagnostic(const_eval_remainder_overflow),
            PointerArithOverflow => {
                handler.struct_diagnostic(const_eval_pointer_arithmetic_overflow)
            }
            InvalidMeta(InvalidMetaKind::SliceTooBig) => {
                handler.struct_diagnostic(const_eval_invalid_meta_slice)
            }
            InvalidMeta(InvalidMetaKind::TooBig) => {
                handler.struct_diagnostic(const_eval_invalid_meta)
            }
            UnterminatedCString(ptr) => {
                let mut builder = handler.struct_diagnostic(const_eval_unterminated_c_string);

                builder.set_arg("pointer", ptr);

                builder
            }
            PointerUseAfterFree(alloc_id, msg) => {
                let mut builder = handler.struct_diagnostic(const_eval_pointer_use_after_free);

                builder.set_arg("allocation", alloc_id);
                builder.set_arg("bad_pointer_message", bad_pointer_message(msg, handler));

                builder
            }
            PointerOutOfBounds { alloc_id, alloc_size, ptr_offset, ptr_size, msg } => {
                let mut builder = if ptr_size == Size::ZERO {
                    handler.struct_diagnostic(const_eval_zst_pointer_out_of_bounds)
                } else {
                    handler.struct_diagnostic(const_eval_pointer_out_of_bounds)
                };

                builder
                    .set_arg("alloc_id", alloc_id)
                    .set_arg("alloc_size", alloc_size.bytes())
                    .set_arg("ptr_offset", ptr_offset)
                    .set_arg("ptr_size", ptr_size.bytes())
                    .set_arg("bad_pointer_message", bad_pointer_message(msg, handler));

                builder
            }
            DanglingIntPointer(ptr, msg) => {
                let mut builder = if ptr == 0 {
                    handler.struct_diagnostic(const_eval_dangling_null_pointer)
                } else {
                    handler.struct_diagnostic(const_eval_dangling_int_pointer)
                };

                if ptr != 0 {
                    builder.set_arg("pointer", format!("{ptr:#x}[noalloc]"));
                }

                builder.set_arg("bad_pointer_message", bad_pointer_message(msg, handler));

                builder
            }
            AlignmentCheckFailed { required, has } => {
                let mut builder = handler.struct_diagnostic(const_eval_alignment_check_failed);

                builder.set_arg("required", required.bytes());
                builder.set_arg("has", has.bytes());

                builder
            }
            WriteToReadOnly(alloc) => {
                let mut builder = handler.struct_diagnostic(const_eval_write_to_read_only);

                builder.set_arg("allocation", alloc);

                builder
            }
            DerefFunctionPointer(alloc) => {
                let mut builder = handler.struct_diagnostic(const_eval_deref_function_pointer);

                builder.set_arg("allocation", alloc);

                builder
            }
            DerefVTablePointer(alloc) => {
                let mut builder = handler.struct_diagnostic(const_eval_deref_vtable_pointer);

                builder.set_arg("allocation", alloc);

                builder
            }
            InvalidBool(b) => {
                let mut builder = handler.struct_diagnostic(const_eval_invalid_bool);

                builder.set_arg("value", format!("{b:02x}"));

                builder
            }
            InvalidChar(c) => {
                let mut builder = handler.struct_diagnostic(const_eval_invalid_char);

                builder.set_arg("value", format!("{c:08x}"));

                builder
            }
            InvalidTag(tag) => {
                let mut builder = handler.struct_diagnostic(const_eval_invalid_tag);

                builder.set_arg("tag", format!("{tag:x}"));

                builder
            }
            InvalidFunctionPointer(ptr) => {
                let mut builder = handler.struct_diagnostic(const_eval_invalid_function_pointer);

                builder.set_arg("pointer", ptr);

                builder
            }
            InvalidVTablePointer(ptr) => {
                let mut builder = handler.struct_diagnostic(const_eval_invalid_vtable_pointer);

                builder.set_arg("pointer", ptr);

                builder
            }
            InvalidStr(err) => {
                let mut builder = handler.struct_diagnostic(const_eval_invalid_str);

                builder.set_arg("err", format!("{err}"));

                builder
            }
            InvalidUninitBytes(None) => {
                handler.struct_diagnostic(const_eval_invalid_uninit_bytes_unknown)
            }
            InvalidUninitBytes(Some((alloc, info))) => {
                let mut builder = handler.struct_diagnostic(const_eval_invalid_uninit_bytes);

                builder.set_arg("alloc", alloc);
                builder.set_arg("access", info.access);
                builder.set_arg("uninit", info.bad);

                builder
            }
            DeadLocal => handler.struct_diagnostic(const_eval_dead_local),
            ScalarSizeMismatch(info) => {
                let mut builder = handler.struct_diagnostic(const_eval_scalar_size_mismatch);

                builder.set_arg("target_size", info.target_size);
                builder.set_arg("data_size", info.data_size);

                builder
            }
            UninhabitedEnumVariantWritten(_) => {
                handler.struct_diagnostic(const_eval_uninhabited_enum_variant_written)
            }
            UninhabitedEnumVariantRead(_) => {
                handler.struct_diagnostic(const_eval_uninhabited_enum_variant_read)
            }
            ValidationError(e) => ValidationErrorInfoExt(e).into_diagnostic(handler),
            Custom(x) => {
                let mut builder = handler.struct_diagnostic((x.msg.clone())());

                (x.add_args)(&mut |name, value| {
                    builder.set_arg(name, value);
                });

                builder
            }
        }
    }
}

pub struct ValidationErrorInfoExt<'tcx>(ValidationErrorInfo<'tcx>);

pub struct ValidationErrorInfoExt2<'tcx> {
    err: ValidationErrorInfo<'tcx>,
    span: Span,
    tcx: TyCtxt<'tcx>,
}

impl AddToDiagnostic for ValidationErrorInfoExt2<'_> {
    fn add_to_diagnostic_with<F>(self, diag: &mut Diagnostic, _: F)
    where
        F: Fn(&mut Diagnostic, SubdiagnosticMessage) -> SubdiagnosticMessage,
    {
        use crate::fluent_generated::*;
        use crate::interpret::ValidationErrorKind::*;

        fn get_range_arg(r: WrappingRange, max_hi: u128, handler: &Handler) -> String {
            let WrappingRange { start: lo, end: hi } = r;
            assert!(hi <= max_hi);
            let msg = if lo > hi {
                fluent::const_eval_range_wrapping
            } else if lo == hi {
                fluent::const_eval_range_singular
            } else if lo == 0 {
                assert!(hi < max_hi, "should not be printing if the range covers everything");
                fluent::const_eval_range_upper
            } else if hi == max_hi {
                assert!(lo > 0, "should not be printing if the range covers everything");
                fluent::const_eval_range_lower
            } else {
                fluent::const_eval_range
            };

            let args = [
                ("lo".into(), DiagnosticArgValue::Str(lo.to_string().into())),
                ("hi".into(), DiagnosticArgValue::Str(hi.to_string().into())),
            ];
            let args = args.iter().map(|(a, b)| (a, b));
            handler.eagerly_translate_to_string(msg, args)
        }

        let handler = &self.tcx.sess.parse_sess.span_diagnostic;

        use crate::fluent_generated as fluent;

        let front_matter = if let Some(path) = self.err.path {
            handler.eagerly_translate_to_string(
                fluent::const_eval_validation_front_matter_invalid_value_with_path,
                [("path".into(), DiagnosticArgValue::Str(path.into()))].iter().map(|(a, b)| (a, b)),
            )
        } else {
            handler.eagerly_translate_to_string(
                fluent::const_eval_validation_front_matter_invalid_value,
                [].into_iter(),
            )
        };

        // Allow untranslatable_diagnostic lint, since we
        // eagerly translate our subdiagnostics by using the
        // eagerly_translate_with_args!() macro.
        #[allow(rustc::untranslatable_diagnostic)]
        match self.err.kind {
            PtrToUninhabited { ptr_kind: PointerKind::Box, ty } => {
                // FIXME(victor-timofei): maybe it would be a good idea to add a
                // span_label_eagerly_translated!() macro
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_box_to_uninhabited,
                    "ty",
                    ty.to_string(),
                    "front_matter",
                    front_matter,
                );

                diag.span_label(self.span, msg);
            }
            PtrToUninhabited { ptr_kind: PointerKind::Ref, ty } => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_ref_to_uninhabited,
                    "ty",
                    ty.to_string(),
                    "front_matter",
                    front_matter,
                );

                diag.span_label(self.span, msg);
            }
            PtrToStatic { ptr_kind: PointerKind::Box, .. } => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_box_to_static,
                    "front_matter",
                    front_matter,
                );

                diag.span_label(self.span, msg);
            }
            PtrToStatic { ptr_kind: PointerKind::Ref, .. } => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_ref_to_static,
                    "front_matter",
                    front_matter,
                );

                diag.span_label(self.span, msg);
            }
            PtrToMut { ptr_kind: PointerKind::Box, .. } => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_box_to_mut,
                    "front_matter",
                    front_matter,
                );

                diag.span_label(self.span, msg);
            }
            PtrToMut { ptr_kind: PointerKind::Ref, .. } => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_ref_to_mut,
                    "front_matter",
                    front_matter,
                );

                diag.span_label(self.span, msg);
            }
            PointerAsInt { expected } => {
                diag.help(const_eval_ptr_as_bytes_1);
                diag.help(const_eval_ptr_as_bytes_2);

                let expected = match expected {
                    ExpectedKind::Reference => fluent::const_eval_validation_expected_ref,
                    ExpectedKind::Box => fluent::const_eval_validation_expected_box,
                    ExpectedKind::RawPtr => fluent::const_eval_validation_expected_raw_ptr,
                    ExpectedKind::InitScalar => fluent::const_eval_validation_expected_init_scalar,
                    ExpectedKind::Bool => fluent::const_eval_validation_expected_bool,
                    ExpectedKind::Char => fluent::const_eval_validation_expected_char,
                    ExpectedKind::Float => fluent::const_eval_validation_expected_float,
                    ExpectedKind::Int => fluent::const_eval_validation_expected_int,
                    ExpectedKind::FnPtr => fluent::const_eval_validation_expected_fn_ptr,
                    ExpectedKind::EnumTag => fluent::const_eval_validation_expected_enum_tag,
                    ExpectedKind::Str => fluent::const_eval_validation_expected_str,
                };
                let expected = handler.eagerly_translate_to_string(expected, [].into_iter());

                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_pointer_as_int,
                    "expected",
                    expected,
                    "front_matter",
                    front_matter,
                );

                diag.span_label(self.span, msg);
            }
            PartialPointer => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_partial_pointer,
                    "front_matter",
                    front_matter,
                );
                diag.span_label(self.span, msg);

                diag.help(const_eval_ptr_as_bytes_1);
                diag.help(const_eval_ptr_as_bytes_2);
            }
            MutableRefInConst => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_mutable_ref_in_const,
                    "front_matter",
                    front_matter,
                );

                diag.span_label(self.span, msg);
            }
            NullFnPtr => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_null_fn_ptr,
                    "front_matter",
                    front_matter,
                );

                diag.span_label(self.span, msg);
            }
            NeverVal => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_never_val,
                    "front_matter",
                    front_matter,
                );

                diag.span_label(self.span, msg);
            }
            NullablePtrOutOfRange { range, max_value } => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_nullable_ptr_out_of_range,
                    "front_matter",
                    front_matter,
                    "in_range",
                    get_range_arg(range, max_value, &handler),
                );

                diag.span_label(self.span, msg);
            }
            PtrOutOfRange { range, max_value } => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_ptr_out_of_range,
                    "front_matter",
                    front_matter,
                    "in_range",
                    get_range_arg(range, max_value, &handler),
                );

                diag.span_label(self.span, msg);
            }
            OutOfRange { range, max_value, value } => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_out_of_range,
                    "front_matter",
                    front_matter,
                    "in_range",
                    get_range_arg(range, max_value, &handler),
                    "value",
                    value,
                );

                diag.span_label(self.span, msg);
            }
            UnsafeCell => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_unsafe_cell,
                    "front_matter",
                    front_matter,
                );

                diag.span_label(self.span, msg);
            }
            UninhabitedVal { ty } => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_uninhabited_val,
                    "front_matter",
                    front_matter,
                    "ty",
                    ty.to_string(),
                );

                diag.span_label(self.span, msg);
            }
            InvalidEnumTag { value } => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_invalid_enum_tag,
                    "front_matter",
                    front_matter,
                    "value",
                    value,
                );

                diag.span_label(self.span, msg);
            }
            UninhabitedEnumVariant => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_uninhabited_enum_variant,
                    "front_matter",
                    front_matter,
                );

                diag.span_label(self.span, msg);
            }
            Uninit { expected } => {
                let expected = match expected {
                    ExpectedKind::Reference => fluent::const_eval_validation_expected_ref,
                    ExpectedKind::Box => fluent::const_eval_validation_expected_box,
                    ExpectedKind::RawPtr => fluent::const_eval_validation_expected_raw_ptr,
                    ExpectedKind::InitScalar => fluent::const_eval_validation_expected_init_scalar,
                    ExpectedKind::Bool => fluent::const_eval_validation_expected_bool,
                    ExpectedKind::Char => fluent::const_eval_validation_expected_char,
                    ExpectedKind::Float => fluent::const_eval_validation_expected_float,
                    ExpectedKind::Int => fluent::const_eval_validation_expected_int,
                    ExpectedKind::FnPtr => fluent::const_eval_validation_expected_fn_ptr,
                    ExpectedKind::EnumTag => fluent::const_eval_validation_expected_enum_tag,
                    ExpectedKind::Str => fluent::const_eval_validation_expected_str,
                };
                let expected = handler.eagerly_translate_to_string(expected, [].into_iter());

                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_uninit,
                    "expected",
                    expected,
                    "front_matter",
                    front_matter,
                );

                diag.span_label(self.span, msg);
            }
            InvalidVTablePtr { value } => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_invalid_vtable_ptr,
                    "front_matter",
                    front_matter,
                    "value",
                    value,
                );

                diag.span_label(self.span, msg);
            }
            InvalidMetaSliceTooLarge { ptr_kind: PointerKind::Box } => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_invalid_box_slice_meta,
                    "front_matter",
                    front_matter,
                );

                diag.span_label(self.span, msg);
            }
            InvalidMetaSliceTooLarge { ptr_kind: PointerKind::Ref } => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_invalid_ref_slice_meta,
                    "front_matter",
                    front_matter,
                );

                diag.span_label(self.span, msg);
            }
            InvalidMetaTooLarge { ptr_kind: PointerKind::Box } => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_invalid_box_meta,
                    "front_matter",
                    front_matter,
                );

                diag.span_label(self.span, msg);
            }
            InvalidMetaTooLarge { ptr_kind: PointerKind::Ref } => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_invalid_ref_meta,
                    "front_matter",
                    front_matter,
                );

                diag.span_label(self.span, msg);
            }
            UnalignedPtr { ptr_kind: PointerKind::Ref, required_bytes, found_bytes } => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_unaligned_ref,
                    "front_matter",
                    front_matter,
                    "required_bytes",
                    required_bytes.to_string(),
                    "found_bytes",
                    found_bytes.to_string(),
                );

                diag.span_label(self.span, msg);
            }
            UnalignedPtr { ptr_kind: PointerKind::Box, required_bytes, found_bytes } => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_unaligned_box,
                    "front_matter",
                    front_matter,
                    "required_bytes",
                    required_bytes.to_string(),
                    "found_bytes",
                    found_bytes.to_string(),
                );

                diag.span_label(self.span, msg);
            }

            NullPtr { ptr_kind: PointerKind::Box } => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_null_box,
                    "front_matter",
                    front_matter,
                );

                diag.span_label(self.span, msg);
            }
            NullPtr { ptr_kind: PointerKind::Ref } => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_null_ref,
                    "front_matter",
                    front_matter,
                );

                diag.span_label(self.span, msg);
            }
            DanglingPtrNoProvenance { ptr_kind: PointerKind::Box, pointer } => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_dangling_box_no_provenance,
                    "front_matter",
                    front_matter,
                    "pointer",
                    pointer,
                );

                diag.span_label(self.span, msg);
            }
            DanglingPtrNoProvenance { ptr_kind: PointerKind::Ref, pointer } => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_dangling_ref_no_provenance,
                    "front_matter",
                    front_matter,
                    "pointer",
                    pointer,
                );

                diag.span_label(self.span, msg);
            }
            DanglingPtrOutOfBounds { ptr_kind: PointerKind::Box } => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_dangling_box_out_of_bounds,
                    "front_matter",
                    front_matter,
                );

                diag.span_label(self.span, msg);
            }
            DanglingPtrOutOfBounds { ptr_kind: PointerKind::Ref } => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_dangling_ref_out_of_bounds,
                    "front_matter",
                    front_matter,
                );

                diag.span_label(self.span, msg);
            }
            DanglingPtrUseAfterFree { ptr_kind: PointerKind::Box } => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_dangling_box_use_after_free,
                    "front_matter",
                    front_matter,
                );

                diag.span_label(self.span, msg);
            }
            DanglingPtrUseAfterFree { ptr_kind: PointerKind::Ref } => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_dangling_ref_use_after_free,
                    "front_matter",
                    front_matter,
                );

                diag.span_label(self.span, msg);
            }
            InvalidBool { value } => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_invalid_bool,
                    "front_matter",
                    front_matter,
                    "value",
                    value,
                );

                diag.span_label(self.span, msg);
            }
            InvalidChar { value } => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_invalid_char,
                    "front_matter",
                    front_matter,
                    "value",
                    value,
                );

                diag.span_label(self.span, msg);
            }
            InvalidFnPtr { value } => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_invalid_fn_ptr,
                    "front_matter",
                    front_matter,
                    "value",
                    value,
                );

                diag.span_label(self.span, msg);
            }
        }
    }
}

impl IntoDiagnostic<'_> for ValidationErrorInfoExt<'_> {
    fn into_diagnostic(self, handler: &'_ Handler) -> DiagnosticBuilder<'_, ErrorGuaranteed> {
        use crate::fluent_generated::*;
        use crate::interpret::ValidationErrorKind::*;

        fn add_range_arg<G: EmissionGuarantee>(
            r: WrappingRange,
            max_hi: u128,
            handler: &Handler,
            err: &mut DiagnosticBuilder<'_, G>,
        ) {
            let WrappingRange { start: lo, end: hi } = r;
            assert!(hi <= max_hi);
            let msg = if lo > hi {
                fluent::const_eval_range_wrapping
            } else if lo == hi {
                fluent::const_eval_range_singular
            } else if lo == 0 {
                assert!(hi < max_hi, "should not be printing if the range covers everything");
                fluent::const_eval_range_upper
            } else if hi == max_hi {
                assert!(lo > 0, "should not be printing if the range covers everything");
                fluent::const_eval_range_lower
            } else {
                fluent::const_eval_range
            };

            let args = [
                ("lo".into(), DiagnosticArgValue::Str(lo.to_string().into())),
                ("hi".into(), DiagnosticArgValue::Str(hi.to_string().into())),
            ];
            let args = args.iter().map(|(a, b)| (a, b));
            let message = handler.eagerly_translate_to_string(msg, args);
            err.set_arg("in_range", message);
        }

        let mut builder = match self.0.kind {
            PtrToUninhabited { ptr_kind: PointerKind::Box, ty } => {
                let mut builder =
                    handler.struct_diagnostic(const_eval_validation_box_to_uninhabited);

                builder.set_arg("ty", ty);

                builder
            }
            PtrToUninhabited { ptr_kind: PointerKind::Ref, ty } => {
                let mut builder =
                    handler.struct_diagnostic(const_eval_validation_ref_to_uninhabited);

                builder.set_arg("ty", ty);

                builder
            }
            PtrToStatic { ptr_kind: PointerKind::Box, .. } => {
                handler.struct_diagnostic(const_eval_validation_box_to_static)
            }
            PtrToStatic { ptr_kind: PointerKind::Ref, .. } => {
                handler.struct_diagnostic(const_eval_validation_ref_to_static)
            }
            PtrToMut { ptr_kind: PointerKind::Box, .. } => {
                handler.struct_diagnostic(const_eval_validation_box_to_mut)
            }
            PtrToMut { ptr_kind: PointerKind::Ref, .. } => {
                handler.struct_diagnostic(const_eval_validation_ref_to_mut)
            }
            PointerAsInt { expected } => {
                let mut builder = handler.struct_diagnostic(const_eval_validation_pointer_as_int);

                builder.help(const_eval_ptr_as_bytes_1);
                builder.help(const_eval_ptr_as_bytes_2);

                let msg = match expected {
                    ExpectedKind::Reference => fluent::const_eval_validation_expected_ref,
                    ExpectedKind::Box => fluent::const_eval_validation_expected_box,
                    ExpectedKind::RawPtr => fluent::const_eval_validation_expected_raw_ptr,
                    ExpectedKind::InitScalar => fluent::const_eval_validation_expected_init_scalar,
                    ExpectedKind::Bool => fluent::const_eval_validation_expected_bool,
                    ExpectedKind::Char => fluent::const_eval_validation_expected_char,
                    ExpectedKind::Float => fluent::const_eval_validation_expected_float,
                    ExpectedKind::Int => fluent::const_eval_validation_expected_int,
                    ExpectedKind::FnPtr => fluent::const_eval_validation_expected_fn_ptr,
                    ExpectedKind::EnumTag => fluent::const_eval_validation_expected_enum_tag,
                    ExpectedKind::Str => fluent::const_eval_validation_expected_str,
                };
                let msg = handler.eagerly_translate_to_string(msg, [].into_iter());
                builder.set_arg("expected", msg);

                builder
            }
            PartialPointer => {
                let mut builder = handler.struct_diagnostic(const_eval_validation_partial_pointer);

                builder.help(const_eval_ptr_as_bytes_1);
                builder.help(const_eval_ptr_as_bytes_2);

                builder
            }
            MutableRefInConst => {
                handler.struct_diagnostic(const_eval_validation_mutable_ref_in_const)
            }
            NullFnPtr => handler.struct_diagnostic(const_eval_validation_null_fn_ptr),
            NeverVal => handler.struct_diagnostic(const_eval_validation_never_val),
            NullablePtrOutOfRange { range, max_value } => {
                let mut builder =
                    handler.struct_diagnostic(const_eval_validation_nullable_ptr_out_of_range);

                add_range_arg(range, max_value, handler, &mut builder);

                builder
            }
            PtrOutOfRange { range, max_value } => {
                let mut builder = handler.struct_diagnostic(const_eval_validation_ptr_out_of_range);

                add_range_arg(range, max_value, handler, &mut builder);

                builder
            }
            OutOfRange { range, max_value, value } => {
                let mut builder = handler.struct_diagnostic(const_eval_validation_out_of_range);

                builder.set_arg("value", value);
                add_range_arg(range, max_value, handler, &mut builder);

                builder
            }
            UnsafeCell => handler.struct_diagnostic(const_eval_validation_unsafe_cell),
            UninhabitedVal { ty } => {
                let mut builder = handler.struct_diagnostic(const_eval_validation_uninhabited_val);

                builder.set_arg("ty", ty);

                builder
            }
            InvalidEnumTag { value } => {
                let mut builder = handler.struct_diagnostic(const_eval_validation_invalid_enum_tag);

                builder.set_arg("value", value);

                builder
            }
            UninhabitedEnumVariant => {
                handler.struct_diagnostic(const_eval_validation_uninhabited_enum_variant)
            }
            Uninit { expected } => {
                let mut builder = handler.struct_diagnostic(const_eval_validation_uninit);

                let msg = match expected {
                    ExpectedKind::Reference => fluent::const_eval_validation_expected_ref,
                    ExpectedKind::Box => fluent::const_eval_validation_expected_box,
                    ExpectedKind::RawPtr => fluent::const_eval_validation_expected_raw_ptr,
                    ExpectedKind::InitScalar => fluent::const_eval_validation_expected_init_scalar,
                    ExpectedKind::Bool => fluent::const_eval_validation_expected_bool,
                    ExpectedKind::Char => fluent::const_eval_validation_expected_char,
                    ExpectedKind::Float => fluent::const_eval_validation_expected_float,
                    ExpectedKind::Int => fluent::const_eval_validation_expected_int,
                    ExpectedKind::FnPtr => fluent::const_eval_validation_expected_fn_ptr,
                    ExpectedKind::EnumTag => fluent::const_eval_validation_expected_enum_tag,
                    ExpectedKind::Str => fluent::const_eval_validation_expected_str,
                };
                let msg = handler.eagerly_translate_to_string(msg, [].into_iter());
                builder.set_arg("expected", msg);

                builder
            }
            InvalidVTablePtr { value } => {
                let mut builder =
                    handler.struct_diagnostic(const_eval_validation_invalid_vtable_ptr);

                builder.set_arg("value", value);

                builder
            }
            InvalidMetaSliceTooLarge { ptr_kind: PointerKind::Box } => {
                handler.struct_diagnostic(const_eval_validation_invalid_box_slice_meta)
            }
            InvalidMetaSliceTooLarge { ptr_kind: PointerKind::Ref } => {
                handler.struct_diagnostic(const_eval_validation_invalid_ref_slice_meta)
            }
            InvalidMetaTooLarge { ptr_kind: PointerKind::Box } => {
                handler.struct_diagnostic(const_eval_validation_invalid_box_meta)
            }
            InvalidMetaTooLarge { ptr_kind: PointerKind::Ref } => {
                handler.struct_diagnostic(const_eval_validation_invalid_ref_meta)
            }
            UnalignedPtr { ptr_kind: PointerKind::Ref, required_bytes, found_bytes } => {
                let mut builder = handler.struct_diagnostic(const_eval_validation_unaligned_ref);

                builder.set_arg("required_bytes", required_bytes);
                builder.set_arg("found_bytes", found_bytes);

                builder
            }
            UnalignedPtr { ptr_kind: PointerKind::Box, required_bytes, found_bytes } => {
                let mut builder = handler.struct_diagnostic(const_eval_validation_unaligned_box);

                builder.set_arg("required_bytes", required_bytes);
                builder.set_arg("found_bytes", found_bytes);

                builder
            }

            NullPtr { ptr_kind: PointerKind::Box } => {
                handler.struct_diagnostic(const_eval_validation_null_box)
            }
            NullPtr { ptr_kind: PointerKind::Ref } => {
                handler.struct_diagnostic(const_eval_validation_null_ref)
            }
            DanglingPtrNoProvenance { ptr_kind: PointerKind::Box, pointer } => {
                let mut builder =
                    handler.struct_diagnostic(const_eval_validation_dangling_box_no_provenance);

                builder.set_arg("pointer", pointer);

                builder
            }
            DanglingPtrNoProvenance { ptr_kind: PointerKind::Ref, pointer } => {
                let mut builder =
                    handler.struct_diagnostic(const_eval_validation_dangling_ref_no_provenance);

                builder.set_arg("pointer", pointer);

                builder
            }
            DanglingPtrOutOfBounds { ptr_kind: PointerKind::Box } => {
                handler.struct_diagnostic(const_eval_validation_dangling_box_out_of_bounds)
            }
            DanglingPtrOutOfBounds { ptr_kind: PointerKind::Ref } => {
                handler.struct_diagnostic(const_eval_validation_dangling_ref_out_of_bounds)
            }
            DanglingPtrUseAfterFree { ptr_kind: PointerKind::Box } => {
                handler.struct_diagnostic(const_eval_validation_dangling_box_use_after_free)
            }
            DanglingPtrUseAfterFree { ptr_kind: PointerKind::Ref } => {
                handler.struct_diagnostic(const_eval_validation_dangling_ref_use_after_free)
            }
            InvalidBool { value } => {
                let mut builder = handler.struct_diagnostic(const_eval_validation_invalid_bool);

                builder.set_arg("value", value);

                builder
            }
            InvalidChar { value } => {
                let mut builder = handler.struct_diagnostic(const_eval_validation_invalid_char);

                builder.set_arg("value", value);

                builder
            }
            InvalidFnPtr { value } => {
                let mut builder = handler.struct_diagnostic(const_eval_validation_invalid_fn_ptr);

                builder.set_arg("value", value);

                builder
            }
        };

        use crate::fluent_generated as fluent;

        let message = if let Some(path) = self.0.path {
            handler.eagerly_translate_to_string(
                fluent::const_eval_validation_front_matter_invalid_value_with_path,
                [("path".into(), DiagnosticArgValue::Str(path.into()))].iter().map(|(a, b)| (a, b)),
            )
        } else {
            handler.eagerly_translate_to_string(
                fluent::const_eval_validation_front_matter_invalid_value,
                [].into_iter(),
            )
        };

        builder.set_arg("front_matter", message);

        builder
    }
}

pub struct UnsupportedExt(UnsupportedOpInfo);

impl IntoDiagnostic<'_> for UnsupportedExt {
    fn into_diagnostic(self, handler: &'_ Handler) -> DiagnosticBuilder<'_, ErrorGuaranteed> {
        use crate::fluent_generated::*;
        use UnsupportedOpInfo::*;
        match self.0 {
            Unsupported(s) => handler.struct_diagnostic(<std::string::String as Into<
                DiagnosticMessage,
            >>::into(s.clone())),
            OverwritePartialPointer(ptr) => {
                let mut builder = handler.struct_diagnostic(const_eval_partial_pointer_overwrite);

                builder.help(const_eval_ptr_as_bytes_1);
                builder.help(const_eval_ptr_as_bytes_2);
                builder.set_arg("ptr", ptr);

                builder
            }
            ReadPartialPointer(ptr) => {
                let mut builder = handler.struct_diagnostic(const_eval_partial_pointer_copy);

                builder.help(const_eval_ptr_as_bytes_1);
                builder.help(const_eval_ptr_as_bytes_2);
                builder.set_arg("ptr", ptr);

                builder
            }
            // `ReadPointerAsInt(Some(info))` is never printed anyway, it only serves as an error to
            // be further processed by validity checking which then turns it into something nice to
            // print. So it's not worth the effort of having diagnostics that can print the `info`.
            ReadPointerAsInt(_) => {
                let mut builder = handler.struct_diagnostic(const_eval_read_pointer_as_int);

                builder.help(const_eval_ptr_as_bytes_1);
                builder.help(const_eval_ptr_as_bytes_2);

                builder
            }
            ThreadLocalStatic(did) => {
                let mut builder = handler.struct_diagnostic(const_eval_thread_local_static);

                builder.set_arg("did", format!("{did:?}"));

                builder
            }
            ReadExternStatic(did) => {
                let mut builder = handler.struct_diagnostic(const_eval_read_extern_static);

                builder.set_arg("did", format!("{did:?}"));

                builder
            }
        }
    }
}

pub struct UnsupportedOpExt2<'tcx> {
    err: UnsupportedOpInfo,
    span: Span,
    tcx: TyCtxt<'tcx>,
}

impl AddToDiagnostic for UnsupportedOpExt2<'_> {
    fn add_to_diagnostic_with<F>(self, diag: &mut Diagnostic, _: F)
    where
        F: Fn(&mut Diagnostic, SubdiagnosticMessage) -> SubdiagnosticMessage,
    {
        use crate::fluent_generated::*;
        use UnsupportedOpInfo::*;

        let handler = &self.tcx.sess.parse_sess.span_diagnostic;

        // Allow untranslatable_diagnostic lint, since we
        // eagerly translate our subdiagnostics by using the
        // eagerly_translate_with_args!() macro.
        #[allow(rustc::untranslatable_diagnostic)]
        match self.err {
            Unsupported(s) => {
                diag.span_label(
                    self.span,
                    <std::string::String as Into<DiagnosticMessage>>::into(s.clone()),
                );
            }
            OverwritePartialPointer(ptr) => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_partial_pointer_overwrite,
                    "ptr",
                    ptr,
                );
                diag.span_label(self.span, msg);

                diag.help(const_eval_ptr_as_bytes_1);
                diag.help(const_eval_ptr_as_bytes_2);
            }
            ReadPartialPointer(ptr) => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_partial_pointer_copy,
                    "ptr",
                    ptr,
                );
                diag.span_label(self.span, msg);

                diag.help(const_eval_ptr_as_bytes_1);
                diag.help(const_eval_ptr_as_bytes_2);
            }
            // `ReadPointerAsInt(Some(info))` is never printed anyway, it only serves as an error to
            // be further processed by validity checking which then turns it into something nice to
            // print. So it's not worth the effort of having diagnostics that can print the `info`.
            ReadPointerAsInt(_) => {
                diag.span_label(self.span, const_eval_read_pointer_as_int);

                diag.help(const_eval_ptr_as_bytes_1);
                diag.help(const_eval_ptr_as_bytes_2);
            }
            ThreadLocalStatic(did) => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_thread_local_static,
                    "did",
                    format!("{did:?}"),
                );
                diag.span_label(self.span, msg);
            }
            ReadExternStatic(did) => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    const_eval_read_extern_static,
                    "did",
                    format!("{did:?}"),
                );
                diag.span_label(self.span, msg);
            }
        }
    }
}

pub struct InterpErrorExt2<'tcx> {
    pub err: InterpError<'tcx>,
    pub span: Span,
    pub tcx: TyCtxt<'tcx>,
}

impl AddToDiagnostic for InterpErrorExt2<'_> {
    fn add_to_diagnostic_with<F>(self, diag: &mut Diagnostic, _: F)
    where
        F: Fn(&mut Diagnostic, SubdiagnosticMessage) -> SubdiagnosticMessage,
    {
        match self.err {
            InterpError::UndefinedBehavior(ub) => {
                UndefinedBehaviorInfoExt2 { err: ub, span: self.span, tcx: self.tcx }
                    .add_to_diagnostic(diag);
            }
            InterpError::Unsupported(e) => {
                UnsupportedOpExt2 { err: e, span: self.span, tcx: self.tcx }
                    .add_to_diagnostic(diag);
            }
            InterpError::InvalidProgram(e) => {
                InvalidProgramInfoExt2 { err: e, span: self.span, tcx: self.tcx }
                    .add_to_diagnostic(diag);
            }
            InterpError::ResourceExhaustion(e) => {
                ResourceExhaustionExt2 { err: e, span: self.span }.add_to_diagnostic(diag);
            }
            InterpError::MachineStop(e) => {
                MachineStopExt2 { err: e, span: self.span }.add_to_diagnostic(diag);
            }
        }
    }
}

impl InterpErrorExt2<'_> {
    pub fn to_string(&self) -> String {
        // FIXME(victor-timofei): implement this
        todo!("InterpErrorExt2::to_string");
    }
}

pub struct InterpErrorExt<'a>(pub InterpError<'a>);

impl IntoDiagnostic<'_> for InterpErrorExt<'_> {
    fn into_diagnostic(self, handler: &'_ Handler) -> DiagnosticBuilder<'_, ErrorGuaranteed> {
        match self.0 {
            InterpError::UndefinedBehavior(ub) => {
                UndefinedBehaviorInfoExt(ub).into_diagnostic(handler)
            }
            InterpError::Unsupported(e) => UnsupportedExt(e).into_diagnostic(handler),
            InterpError::InvalidProgram(e) => InvalidProgramInfoExt(e).into_diagnostic(handler),
            InterpError::ResourceExhaustion(e) => ResourceExhaustionExt(e).into_diagnostic(handler),
            InterpError::MachineStop(e) => MachineStopExt(e).into_diagnostic(handler),
        }
    }
}

impl InterpErrorExt<'_> {
    /// Translate InterpError to String.
    ///
    /// This should not be used for any user-facing diagnostics,
    /// only for debug messages in the docs.
    pub fn to_string(self) -> String {
        ty::tls::with(move |tcx| {
            let handler = &tcx.sess.parse_sess.span_diagnostic;
            let builder = self.into_diagnostic(handler);
            let s = handler.eagerly_translate_to_string(
                builder.message.first().unwrap().0.to_owned(),
                builder.args(),
            );
            builder.cancel();
            s
        })
    }
}

pub struct MachineStopExt2 {
    err: Box<dyn MachineStopType>,
    span: Span,
}

impl AddToDiagnostic for MachineStopExt2 {
    fn add_to_diagnostic_with<F>(self, diag: &mut Diagnostic, _: F)
    where
        F: Fn(&mut Diagnostic, SubdiagnosticMessage) -> SubdiagnosticMessage,
    {
        diag.span_label(self.span, self.err.diagnostic_message().clone());

        self.err.add_args(&mut |name, value| {
            diag.set_arg(name, value);
        });
    }
}

pub struct MachineStopExt(Box<dyn MachineStopType>);

impl IntoDiagnostic<'_> for MachineStopExt {
    fn into_diagnostic(self, handler: &'_ Handler) -> DiagnosticBuilder<'_, ErrorGuaranteed> {
        let mut builder = handler.struct_diagnostic(self.0.diagnostic_message().clone());

        self.0.add_args(&mut |name, value| {
            builder.set_arg(name, value);
        });
        builder
    }
}

pub struct InvalidProgramInfoExt2<'tcx> {
    err: InvalidProgramInfo<'tcx>,
    span: Span,
    tcx: TyCtxt<'tcx>,
}

impl AddToDiagnostic for InvalidProgramInfoExt2<'_> {
    fn add_to_diagnostic_with<F>(self, diag: &mut Diagnostic, _: F)
    where
        F: Fn(&mut Diagnostic, SubdiagnosticMessage) -> SubdiagnosticMessage,
    {
        use crate::fluent_generated::*;
        let handler = &self.tcx.sess.parse_sess.span_diagnostic;

        // Allow untranslatable_diagnostic lint, since we
        // eagerly translate our subdiagnostics by using the
        // eagerly_translate_with_args!() macro.
        #[allow(rustc::untranslatable_diagnostic)]
        match self.err {
            InvalidProgramInfo::TooGeneric => {
                diag.span_label(self.span, const_eval_too_generic);
            }
            InvalidProgramInfo::AlreadyReported(_) => {
                diag.span_label(self.span, const_eval_already_reported);
            }
            InvalidProgramInfo::Layout(e) => {
                let builder: DiagnosticBuilder<'_, ()> =
                    e.into_diagnostic().into_diagnostic(handler);
                for (name, val) in builder.args() {
                    diag.set_arg(name.clone(), val.clone());
                }
                let msg =
                    handler.eagerly_translate_to_string(e.diagnostic_message(), builder.args());
                builder.cancel();

                diag.span_label(self.span, msg);
            }
            InvalidProgramInfo::FnAbiAdjustForForeignAbi(
                AdjustForForeignAbiError::Unsupported { arch, abi },
            ) => {
                let msg = eagerly_translate_with_args!(
                    handler,
                    rustc_middle::error::middle_adjust_for_foreign_abi_error,
                    "arch",
                    arch.to_ident_string(),
                    "abi",
                    abi.name(),
                );

                diag.span_label(self.span, msg);
            }
            InvalidProgramInfo::ConstPropNonsense => {
                panic!("We had const-prop nonsense, this should never be printed")
            }
        }
    }
}

pub struct InvalidProgramInfoExt<'a>(InvalidProgramInfo<'a>);

impl IntoDiagnostic<'_> for InvalidProgramInfoExt<'_> {
    fn into_diagnostic(self, handler: &'_ Handler) -> DiagnosticBuilder<'_, ErrorGuaranteed> {
        use crate::fluent_generated::*;
        match self.0 {
            InvalidProgramInfo::TooGeneric => handler.struct_diagnostic(const_eval_too_generic),
            InvalidProgramInfo::AlreadyReported(_) => {
                handler.struct_diagnostic(const_eval_already_reported)
            }
            InvalidProgramInfo::Layout(e) => {
                let mut builder = handler.struct_diagnostic(e.diagnostic_message());

                let diag: DiagnosticBuilder<'_, ()> = e.into_diagnostic().into_diagnostic(handler);
                for (name, val) in diag.args() {
                    builder.set_arg(name.clone(), val.clone());
                }
                diag.cancel();

                builder
            }
            InvalidProgramInfo::FnAbiAdjustForForeignAbi(
                AdjustForForeignAbiError::Unsupported { arch, abi },
            ) => {
                let mut builder = handler
                    .struct_diagnostic(rustc_middle::error::middle_adjust_for_foreign_abi_error);

                builder.set_arg("arch", arch);
                builder.set_arg("abi", abi.name());

                builder
            }
            InvalidProgramInfo::ConstPropNonsense => {
                panic!("We had const-prop nonsense, this should never be printed")
            }
        }
    }
}

pub struct ResourceExhaustionExt2 {
    err: ResourceExhaustionInfo,
    span: Span,
}

impl AddToDiagnostic for ResourceExhaustionExt2 {
    fn add_to_diagnostic_with<F>(self, diag: &mut Diagnostic, _: F)
    where
        F: Fn(&mut Diagnostic, SubdiagnosticMessage) -> SubdiagnosticMessage,
    {
        use crate::fluent_generated::*;
        let msg = match self.err {
            ResourceExhaustionInfo::StackFrameLimitReached => const_eval_stack_frame_limit_reached,
            ResourceExhaustionInfo::MemoryExhausted => const_eval_memory_exhausted,
            ResourceExhaustionInfo::AddressSpaceFull => const_eval_address_space_full,
        };
        diag.span_label(self.span, msg);
    }
}

pub struct ResourceExhaustionExt(ResourceExhaustionInfo);

impl IntoDiagnostic<'_> for ResourceExhaustionExt {
    fn into_diagnostic(self, handler: &'_ Handler) -> DiagnosticBuilder<'_, ErrorGuaranteed> {
        use crate::fluent_generated::*;
        let msg = match self.0 {
            ResourceExhaustionInfo::StackFrameLimitReached => const_eval_stack_frame_limit_reached,
            ResourceExhaustionInfo::MemoryExhausted => const_eval_memory_exhausted,
            ResourceExhaustionInfo::AddressSpaceFull => const_eval_address_space_full,
        };
        handler.struct_diagnostic(msg)
    }
}
