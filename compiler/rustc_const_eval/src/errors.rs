use rustc_errors::{
    AddToDiagnostic, Diagnostic, DiagnosticArgValue, DiagnosticBuilder, DiagnosticMessage, Handler,
    IntoDiagnostic, IntoDiagnosticArg, SubdiagnosticMessage,
};
use rustc_hir::ConstContext;
use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};
use rustc_middle::mir::interpret::{
    CheckInAllocMsg, ExpectedKind, InterpError, InvalidMetaKind, InvalidProgramInfo,
    MachineStopType, PointerKind, ResourceExhaustionInfo, UndefinedBehaviorInfo, UnsupportedOpInfo,
    ValidationErrorInfo,
};
use rustc_middle::ty::{Ty, TyCtxt};
use rustc_span::Span;
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

struct UndefinedBehaviorInfoExt<'tcx> {
    err: UndefinedBehaviorInfo<'tcx>,
    span: Span,
    tcx: TyCtxt<'tcx>,
}

impl UndefinedBehaviorInfoExt<'_> {
    fn eagerly_translate(self) -> String {
        use crate::fluent_generated::*;
        use UndefinedBehaviorInfo::*;

        let handler = &self.tcx.sess.parse_sess.span_diagnostic;

        match self.err {
            #[allow(rustc::untranslatable_diagnostic)]
            Ub(str) => {
                eagerly_translate_with_args!(handler, str.clone().into(),)
            }
            Unreachable => {
                eagerly_translate_with_args!(handler, const_eval_unreachable,)
            }
            BoundsCheckFailed { len, index } => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_bounds_check_failed,
                    "len",
                    len,
                    "index",
                    index,
                )
            }
            DivisionByZero => {
                eagerly_translate_with_args!(handler, const_eval_division_by_zero,)
            }
            RemainderByZero => {
                eagerly_translate_with_args!(handler, const_eval_remainder_by_zero,)
            }
            DivisionOverflow => {
                eagerly_translate_with_args!(handler, const_eval_division_overflow,)
            }
            RemainderOverflow => {
                eagerly_translate_with_args!(handler, const_eval_remainder_overflow,)
            }
            PointerArithOverflow => {
                eagerly_translate_with_args!(handler, const_eval_pointer_arithmetic_overflow,)
            }
            InvalidMeta(InvalidMetaKind::SliceTooBig) => {
                eagerly_translate_with_args!(handler, const_eval_invalid_meta_slice,)
            }
            InvalidMeta(InvalidMetaKind::TooBig) => {
                eagerly_translate_with_args!(handler, const_eval_invalid_meta,)
            }
            UnterminatedCString(ptr) => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_unterminated_c_string,
                    "pointer",
                    ptr,
                )
            }
            PointerUseAfterFree(alloc_id, msg) => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_pointer_use_after_free,
                    "alloc_id",
                    alloc_id,
                    "bad_pointer_message",
                    bad_pointer_message(msg, handler)
                )
            }
            PointerOutOfBounds { alloc_id, alloc_size, ptr_offset, ptr_size, msg } => {
                let diagnostic_message = if ptr_size == Size::ZERO {
                    const_eval_zst_pointer_out_of_bounds
                } else {
                    const_eval_pointer_out_of_bounds
                };

                eagerly_translate_with_args!(
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
                )
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

                handler.eagerly_translate_to_string(
                    diagnostic_message,
                    args.iter().map(|(a, b)| (a, b)),
                )
            }
            AlignmentCheckFailed { required, has } => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_alignment_check_failed,
                    "required",
                    required.bytes(),
                    "has",
                    has.bytes(),
                )
            }
            WriteToReadOnly(alloc) => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_write_to_read_only,
                    "allocation",
                    alloc,
                )
            }
            DerefFunctionPointer(alloc) => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_deref_function_pointer,
                    "allocation",
                    alloc,
                )
            }
            DerefVTablePointer(alloc) => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_deref_vtable_pointer,
                    "allocation",
                    alloc,
                )
            }
            InvalidBool(b) => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_invalid_bool,
                    "value",
                    format!("{b:02x}"),
                )
            }
            InvalidChar(c) => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_invalid_char,
                    "value",
                    format!("{c:08x}"),
                )
            }
            InvalidTag(tag) => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_invalid_tag,
                    "tag",
                    format!("{tag:x}"),
                )
            }
            InvalidFunctionPointer(ptr) => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_invalid_function_pointer,
                    "pointer",
                    ptr,
                )
            }
            InvalidVTablePointer(ptr) => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_invalid_vtable_pointer,
                    "pointer",
                    ptr,
                )
            }
            InvalidStr(err) => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_invalid_str,
                    "err",
                    format!("{err}"),
                )
            }
            InvalidUninitBytes(None) => {
                eagerly_translate_with_args!(handler, const_eval_invalid_uninit_bytes_unknown,)
            }
            InvalidUninitBytes(Some((alloc, info))) => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_invalid_uninit_bytes,
                    "alloc",
                    alloc,
                    "access",
                    info.access,
                    "uninit",
                    info.bad,
                )
            }
            DeadLocal => {
                eagerly_translate_with_args!(handler, const_eval_dead_local,)
            }
            ScalarSizeMismatch(info) => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_scalar_size_mismatch,
                    "target_size",
                    info.target_size,
                    "data_size",
                    info.data_size,
                )
            }
            UninhabitedEnumVariantWritten(_) => {
                eagerly_translate_with_args!(handler, const_eval_uninhabited_enum_variant_written,)
            }
            UninhabitedEnumVariantRead(_) => {
                eagerly_translate_with_args!(handler, const_eval_uninhabited_enum_variant_read,)
            }
            ValidationError(err) => {
                ValidationErrorInfoExt { err, tcx: self.tcx, span: self.span }.eagerly_translate()
            }
            Custom(x) => {
                let mut args = Vec::new();

                (x.add_args)(&mut |name, value| {
                    args.push((name, value));
                });

                handler.eagerly_translate_to_string(
                    (x.msg.clone())(),
                    args.iter().map(|(a, b)| (a, b)),
                )
            }
        }
    }
}

impl AddToDiagnostic for UndefinedBehaviorInfoExt<'_> {
    fn add_to_diagnostic_with<F>(self, diag: &mut Diagnostic, _: F)
    where
        F: Fn(&mut Diagnostic, SubdiagnosticMessage) -> SubdiagnosticMessage,
    {
        use UndefinedBehaviorInfo::*;

        match self.err {
            Ub(_)
            | Unreachable
            | BoundsCheckFailed { .. }
            | DivisionByZero
            | RemainderByZero
            | DivisionOverflow
            | RemainderOverflow
            | PointerArithOverflow
            | InvalidMeta(_)
            | UnterminatedCString(_)
            | PointerUseAfterFree(_, _)
            | PointerOutOfBounds { .. }
            | DanglingIntPointer(_, _)
            | AlignmentCheckFailed { .. }
            | WriteToReadOnly(_)
            | DerefFunctionPointer(_)
            | DerefVTablePointer(_)
            | InvalidBool(_)
            | InvalidChar(_)
            | InvalidTag(_)
            | InvalidFunctionPointer(_)
            | InvalidVTablePointer(_)
            | InvalidStr(_)
            | InvalidUninitBytes(_)
            | DeadLocal
            | ScalarSizeMismatch(_)
            | UninhabitedEnumVariantWritten(_)
            | UninhabitedEnumVariantRead(_)
            | Custom(_) => {
                #[allow(rustc::untranslatable_diagnostic)]
                diag.span_label(self.span, self.eagerly_translate());
            }
            ValidationError(err) => {
                ValidationErrorInfoExt { err, tcx: self.tcx, span: self.span }
                    .add_to_diagnostic(diag);
            }
        }
    }
}

struct ValidationErrorInfoExt<'tcx> {
    err: ValidationErrorInfo<'tcx>,
    span: Span,
    tcx: TyCtxt<'tcx>,
}

impl ValidationErrorInfoExt<'_> {
    fn eagerly_translate(self) -> String {
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

        match self.err.kind {
            PtrToUninhabited { ptr_kind: PointerKind::Box, ty } => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_box_to_uninhabited,
                    "ty",
                    ty.to_string(),
                    "front_matter",
                    front_matter,
                )
            }
            PtrToUninhabited { ptr_kind: PointerKind::Ref, ty } => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_ref_to_uninhabited,
                    "ty",
                    ty.to_string(),
                    "front_matter",
                    front_matter,
                )
            }
            PtrToStatic { ptr_kind: PointerKind::Box, .. } => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_box_to_static,
                    "front_matter",
                    front_matter,
                )
            }
            PtrToStatic { ptr_kind: PointerKind::Ref, .. } => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_ref_to_static,
                    "front_matter",
                    front_matter,
                )
            }
            PtrToMut { ptr_kind: PointerKind::Box, .. } => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_box_to_mut,
                    "front_matter",
                    front_matter,
                )
            }
            PtrToMut { ptr_kind: PointerKind::Ref, .. } => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_ref_to_mut,
                    "front_matter",
                    front_matter,
                )
            }
            PointerAsInt { expected } => {
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

                eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_pointer_as_int,
                    "expected",
                    expected,
                    "front_matter",
                    front_matter,
                )
            }
            PartialPointer => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_partial_pointer,
                    "front_matter",
                    front_matter,
                )
            }
            MutableRefInConst => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_mutable_ref_in_const,
                    "front_matter",
                    front_matter,
                )
            }
            NullFnPtr => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_null_fn_ptr,
                    "front_matter",
                    front_matter,
                )
            }
            NeverVal => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_never_val,
                    "front_matter",
                    front_matter,
                )
            }
            NullablePtrOutOfRange { range, max_value } => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_nullable_ptr_out_of_range,
                    "front_matter",
                    front_matter,
                    "in_range",
                    get_range_arg(range, max_value, &handler),
                )
            }
            PtrOutOfRange { range, max_value } => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_ptr_out_of_range,
                    "front_matter",
                    front_matter,
                    "in_range",
                    get_range_arg(range, max_value, &handler),
                )
            }
            OutOfRange { range, max_value, value } => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_out_of_range,
                    "front_matter",
                    front_matter,
                    "in_range",
                    get_range_arg(range, max_value, &handler),
                    "value",
                    value,
                )
            }
            UnsafeCell => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_unsafe_cell,
                    "front_matter",
                    front_matter,
                )
            }
            UninhabitedVal { ty } => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_uninhabited_val,
                    "front_matter",
                    front_matter,
                    "ty",
                    ty.to_string(),
                )
            }
            InvalidEnumTag { value } => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_invalid_enum_tag,
                    "front_matter",
                    front_matter,
                    "value",
                    value,
                )
            }
            UninhabitedEnumVariant => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_uninhabited_enum_variant,
                    "front_matter",
                    front_matter,
                )
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

                eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_uninit,
                    "expected",
                    expected,
                    "front_matter",
                    front_matter,
                )
            }
            InvalidVTablePtr { value } => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_invalid_vtable_ptr,
                    "front_matter",
                    front_matter,
                    "value",
                    value,
                )
            }
            InvalidMetaSliceTooLarge { ptr_kind: PointerKind::Box } => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_invalid_box_slice_meta,
                    "front_matter",
                    front_matter,
                )
            }
            InvalidMetaSliceTooLarge { ptr_kind: PointerKind::Ref } => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_invalid_ref_slice_meta,
                    "front_matter",
                    front_matter,
                )
            }
            InvalidMetaTooLarge { ptr_kind: PointerKind::Box } => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_invalid_box_meta,
                    "front_matter",
                    front_matter,
                )
            }
            InvalidMetaTooLarge { ptr_kind: PointerKind::Ref } => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_invalid_ref_meta,
                    "front_matter",
                    front_matter,
                )
            }
            UnalignedPtr { ptr_kind: PointerKind::Ref, required_bytes, found_bytes } => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_unaligned_ref,
                    "front_matter",
                    front_matter,
                    "required_bytes",
                    required_bytes.to_string(),
                    "found_bytes",
                    found_bytes.to_string(),
                )
            }
            UnalignedPtr { ptr_kind: PointerKind::Box, required_bytes, found_bytes } => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_unaligned_box,
                    "front_matter",
                    front_matter,
                    "required_bytes",
                    required_bytes.to_string(),
                    "found_bytes",
                    found_bytes.to_string(),
                )
            }

            NullPtr { ptr_kind: PointerKind::Box } => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_null_box,
                    "front_matter",
                    front_matter,
                )
            }
            NullPtr { ptr_kind: PointerKind::Ref } => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_null_ref,
                    "front_matter",
                    front_matter,
                )
            }
            DanglingPtrNoProvenance { ptr_kind: PointerKind::Box, pointer } => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_dangling_box_no_provenance,
                    "front_matter",
                    front_matter,
                    "pointer",
                    pointer,
                )
            }
            DanglingPtrNoProvenance { ptr_kind: PointerKind::Ref, pointer } => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_dangling_ref_no_provenance,
                    "front_matter",
                    front_matter,
                    "pointer",
                    pointer,
                )
            }
            DanglingPtrOutOfBounds { ptr_kind: PointerKind::Box } => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_dangling_box_out_of_bounds,
                    "front_matter",
                    front_matter,
                )
            }
            DanglingPtrOutOfBounds { ptr_kind: PointerKind::Ref } => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_dangling_ref_out_of_bounds,
                    "front_matter",
                    front_matter,
                )
            }
            DanglingPtrUseAfterFree { ptr_kind: PointerKind::Box } => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_dangling_box_use_after_free,
                    "front_matter",
                    front_matter,
                )
            }
            DanglingPtrUseAfterFree { ptr_kind: PointerKind::Ref } => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_dangling_ref_use_after_free,
                    "front_matter",
                    front_matter,
                )
            }
            InvalidBool { value } => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_invalid_bool,
                    "front_matter",
                    front_matter,
                    "value",
                    value,
                )
            }
            InvalidChar { value } => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_invalid_char,
                    "front_matter",
                    front_matter,
                    "value",
                    value,
                )
            }
            InvalidFnPtr { value } => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_validation_invalid_fn_ptr,
                    "front_matter",
                    front_matter,
                    "value",
                    value,
                )
            }
        }
    }
}

impl AddToDiagnostic for ValidationErrorInfoExt<'_> {
    fn add_to_diagnostic_with<F>(self, diag: &mut Diagnostic, _: F)
    where
        F: Fn(&mut Diagnostic, SubdiagnosticMessage) -> SubdiagnosticMessage,
    {
        use crate::fluent_generated::*;
        use crate::interpret::ValidationErrorKind::*;

        match self.err.kind {
            PtrToUninhabited { .. }
            | PtrToStatic { .. }
            | PtrToMut { .. }
            | MutableRefInConst
            | NullFnPtr
            | NeverVal
            | NullablePtrOutOfRange { .. }
            | PtrOutOfRange { .. }
            | OutOfRange { .. }
            | UnsafeCell
            | UninhabitedVal { .. }
            | InvalidEnumTag { .. }
            | UninhabitedEnumVariant
            | Uninit { .. }
            | InvalidVTablePtr { .. }
            | InvalidMetaSliceTooLarge { .. }
            | InvalidMetaTooLarge { .. }
            | UnalignedPtr { .. }
            | NullPtr { .. }
            | DanglingPtrNoProvenance { .. }
            | DanglingPtrOutOfBounds { .. }
            | DanglingPtrUseAfterFree { .. }
            | InvalidBool { .. }
            | InvalidChar { .. }
            | InvalidFnPtr { .. } => {}
            PointerAsInt { .. } | PartialPointer => {
                diag.help(const_eval_ptr_as_bytes_1);
                diag.help(const_eval_ptr_as_bytes_2);
            }
        }

        #[allow(rustc::untranslatable_diagnostic)]
        diag.span_label(self.span, self.eagerly_translate());
    }
}

struct UnsupportedOpExt<'tcx> {
    err: UnsupportedOpInfo,
    span: Span,
    tcx: TyCtxt<'tcx>,
}

impl UnsupportedOpExt<'_> {
    fn eagerly_translate(self) -> String {
        use crate::fluent_generated::*;
        use UnsupportedOpInfo::*;

        let handler = &self.tcx.sess.parse_sess.span_diagnostic;

        match self.err {
            Unsupported(s) => {
                eagerly_translate_with_args!(
                    handler,
                    <std::string::String as Into<DiagnosticMessage>>::into(s.clone()),
                )
            }
            OverwritePartialPointer(ptr) => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_partial_pointer_overwrite,
                    "ptr",
                    ptr,
                )
            }
            ReadPartialPointer(ptr) => {
                eagerly_translate_with_args!(handler, const_eval_partial_pointer_copy, "ptr", ptr,)
            }
            // `ReadPointerAsInt(Some(info))` is never printed anyway, it only serves as an error to
            // be further processed by validity checking which then turns it into something nice to
            // print. So it's not worth the effort of having diagnostics that can print the `info`.
            ReadPointerAsInt(_) => {
                eagerly_translate_with_args!(handler, const_eval_read_pointer_as_int,)
            }
            ThreadLocalStatic(did) => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_thread_local_static,
                    "did",
                    format!("{did:?}"),
                )
            }
            ReadExternStatic(did) => {
                eagerly_translate_with_args!(
                    handler,
                    const_eval_read_extern_static,
                    "did",
                    format!("{did:?}"),
                )
            }
        }
    }
}

impl AddToDiagnostic for UnsupportedOpExt<'_> {
    fn add_to_diagnostic_with<F>(self, diag: &mut Diagnostic, _: F)
    where
        F: Fn(&mut Diagnostic, SubdiagnosticMessage) -> SubdiagnosticMessage,
    {
        use crate::fluent_generated::*;
        use UnsupportedOpInfo::*;

        match self.err {
            Unsupported(_) | ThreadLocalStatic(_) | ReadExternStatic(_) => {}
            OverwritePartialPointer(_) | ReadPartialPointer(_) | ReadPointerAsInt(_) => {
                diag.help(const_eval_ptr_as_bytes_1);
                diag.help(const_eval_ptr_as_bytes_2);
            }
        }

        #[allow(rustc::untranslatable_diagnostic)]
        diag.span_label(self.span, self.eagerly_translate());
    }
}

pub struct InterpErrorExt<'tcx> {
    pub err: InterpError<'tcx>,
    pub span: Span,
    pub tcx: TyCtxt<'tcx>,
}

impl AddToDiagnostic for InterpErrorExt<'_> {
    fn add_to_diagnostic_with<F>(self, diag: &mut Diagnostic, _: F)
    where
        F: Fn(&mut Diagnostic, SubdiagnosticMessage) -> SubdiagnosticMessage,
    {
        match self.err {
            InterpError::UndefinedBehavior(ub) => {
                UndefinedBehaviorInfoExt { err: ub, span: self.span, tcx: self.tcx }
                    .add_to_diagnostic(diag);
            }
            InterpError::Unsupported(e) => {
                UnsupportedOpExt { err: e, span: self.span, tcx: self.tcx }.add_to_diagnostic(diag);
            }
            InterpError::InvalidProgram(e) => {
                InvalidProgramInfoExt { err: e, span: self.span, tcx: self.tcx }
                    .add_to_diagnostic(diag);
            }
            InterpError::ResourceExhaustion(e) => {
                ResourceExhaustionExt { err: e, span: self.span, tcx: self.tcx }
                    .add_to_diagnostic(diag);
            }
            InterpError::MachineStop(e) => {
                MachineStopExt { err: e, span: self.span, tcx: self.tcx }.add_to_diagnostic(diag);
            }
        }
    }
}

impl InterpErrorExt<'_> {
    /// Translate InterpError to String.
    ///
    /// This should not be used for any user-facing diagnostics,
    /// only for debug messages in the docs.
    pub fn to_string(self) -> String {
        // FIXME(victor-timofei): implement this
        match self.err {
            InterpError::UndefinedBehavior(ub) => {
                UndefinedBehaviorInfoExt { err: ub, span: self.span, tcx: self.tcx }
                    .eagerly_translate()
            }
            InterpError::Unsupported(e) => {
                UnsupportedOpExt { err: e, span: self.span, tcx: self.tcx }.eagerly_translate()
            }
            InterpError::InvalidProgram(e) => {
                InvalidProgramInfoExt { err: e, span: self.span, tcx: self.tcx }.eagerly_translate()
            }
            InterpError::ResourceExhaustion(e) => {
                ResourceExhaustionExt { err: e, span: self.span, tcx: self.tcx }.eagerly_translate()
            }
            InterpError::MachineStop(e) => {
                MachineStopExt { err: e, span: self.span, tcx: self.tcx }.eagerly_translate()
            }
        }
    }
}

struct MachineStopExt<'tcx> {
    err: Box<dyn MachineStopType>,
    span: Span,
    tcx: TyCtxt<'tcx>,
}

impl MachineStopExt<'_> {
    fn eagerly_translate(self) -> String {
        let mut args = Vec::new();
        let msg = self.err.diagnostic_message().clone();
        self.err.add_args(&mut |name, value| {
            args.push((name, value));
        });

        let handler = &self.tcx.sess.parse_sess.span_diagnostic;

        handler.eagerly_translate_to_string(msg, args.iter().map(|(a, b)| (a, b)))
    }
}

impl AddToDiagnostic for MachineStopExt<'_> {
    fn add_to_diagnostic_with<F>(self, diag: &mut Diagnostic, _: F)
    where
        F: Fn(&mut Diagnostic, SubdiagnosticMessage) -> SubdiagnosticMessage,
    {
        #[allow(rustc::untranslatable_diagnostic)]
        diag.span_label(self.span, self.eagerly_translate());
    }
}

struct InvalidProgramInfoExt<'tcx> {
    err: InvalidProgramInfo<'tcx>,
    span: Span,
    tcx: TyCtxt<'tcx>,
}

impl InvalidProgramInfoExt<'_> {
    fn eagerly_translate(self) -> String {
        use crate::fluent_generated::*;
        let handler = &self.tcx.sess.parse_sess.span_diagnostic;

        match self.err {
            InvalidProgramInfo::TooGeneric => {
                eagerly_translate_with_args!(handler, const_eval_too_generic,)
            }
            InvalidProgramInfo::AlreadyReported(_) => {
                eagerly_translate_with_args!(handler, const_eval_already_reported,)
            }
            InvalidProgramInfo::Layout(e) => {
                let builder: DiagnosticBuilder<'_, ()> =
                    e.into_diagnostic().into_diagnostic(handler);

                let msg =
                    handler.eagerly_translate_to_string(e.diagnostic_message(), builder.args());
                builder.cancel();

                msg
            }
            InvalidProgramInfo::FnAbiAdjustForForeignAbi(
                AdjustForForeignAbiError::Unsupported { arch, abi },
            ) => {
                eagerly_translate_with_args!(
                    handler,
                    rustc_middle::error::middle_adjust_for_foreign_abi_error,
                    "arch",
                    arch.to_ident_string(),
                    "abi",
                    abi.name(),
                )
            }
            InvalidProgramInfo::ConstPropNonsense => {
                panic!("We had const-prop nonsense, this should never be printed")
            }
        }
    }
}

impl AddToDiagnostic for InvalidProgramInfoExt<'_> {
    fn add_to_diagnostic_with<F>(self, diag: &mut Diagnostic, _: F)
    where
        F: Fn(&mut Diagnostic, SubdiagnosticMessage) -> SubdiagnosticMessage,
    {
        match self.err {
            InvalidProgramInfo::TooGeneric
            | InvalidProgramInfo::AlreadyReported(_)
            | InvalidProgramInfo::Layout(_)
            | InvalidProgramInfo::FnAbiAdjustForForeignAbi(_) => {}
            InvalidProgramInfo::ConstPropNonsense => {
                panic!("We had const-prop nonsense, this should never be printed")
            }
        }

        #[allow(rustc::untranslatable_diagnostic)]
        diag.span_label(self.span, self.eagerly_translate());
    }
}

struct ResourceExhaustionExt<'tcx> {
    err: ResourceExhaustionInfo,
    span: Span,
    tcx: TyCtxt<'tcx>,
}

impl ResourceExhaustionExt<'_> {
    fn eagerly_translate(self) -> String {
        let handler = &self.tcx.sess.parse_sess.span_diagnostic;

        use crate::fluent_generated::*;
        let msg = match self.err {
            ResourceExhaustionInfo::StackFrameLimitReached => const_eval_stack_frame_limit_reached,
            ResourceExhaustionInfo::MemoryExhausted => const_eval_memory_exhausted,
            ResourceExhaustionInfo::AddressSpaceFull => const_eval_address_space_full,
        };

        eagerly_translate_with_args!(handler, msg)
    }
}

impl AddToDiagnostic for ResourceExhaustionExt<'_> {
    fn add_to_diagnostic_with<F>(self, diag: &mut Diagnostic, _: F)
    where
        F: Fn(&mut Diagnostic, SubdiagnosticMessage) -> SubdiagnosticMessage,
    {
        #[allow(rustc::untranslatable_diagnostic)]
        diag.span_label(self.span, self.eagerly_translate());
    }
}
