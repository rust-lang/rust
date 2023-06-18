use rustc_errors::{
    DiagnosticArgValue, DiagnosticBuilder, DiagnosticMessage, EmissionGuarantee, Handler,
    IntoDiagnostic,
};
use rustc_hir::ConstContext;
use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};
use rustc_middle::mir::interpret::{
    CheckInAllocMsg, ExpectedKind, InterpError, InvalidMetaKind, InvalidProgramInfo, PointerKind,
    ResourceExhaustionInfo, UndefinedBehaviorInfo, UnsupportedOpInfo, ValidationErrorInfo,
};
use rustc_middle::ty::{self, Ty};
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

pub trait ReportErrorExt {
    /// Returns the diagnostic message for this error.
    fn diagnostic_message(&self) -> DiagnosticMessage;
    fn add_args<G: EmissionGuarantee>(
        self,
        handler: &Handler,
        builder: &mut DiagnosticBuilder<'_, G>,
    );

    fn debug(self) -> String
    where
        Self: Sized,
    {
        ty::tls::with(move |tcx| {
            let mut builder = tcx.sess.struct_allow(DiagnosticMessage::Str(String::new().into()));
            let handler = &tcx.sess.parse_sess.span_diagnostic;
            let message = self.diagnostic_message();
            self.add_args(handler, &mut builder);
            let s = handler.eagerly_translate_to_string(message, builder.args());
            builder.cancel();
            s
        })
    }
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

impl<'a> ReportErrorExt for UndefinedBehaviorInfo<'a> {
    fn diagnostic_message(&self) -> DiagnosticMessage {
        use crate::fluent_generated::*;
        use UndefinedBehaviorInfo::*;
        match self {
            Ub(msg) => msg.clone().into(),
            Unreachable => const_eval_unreachable,
            BoundsCheckFailed { .. } => const_eval_bounds_check_failed,
            DivisionByZero => const_eval_division_by_zero,
            RemainderByZero => const_eval_remainder_by_zero,
            DivisionOverflow => const_eval_division_overflow,
            RemainderOverflow => const_eval_remainder_overflow,
            PointerArithOverflow => const_eval_pointer_arithmetic_overflow,
            InvalidMeta(InvalidMetaKind::SliceTooBig) => const_eval_invalid_meta_slice,
            InvalidMeta(InvalidMetaKind::TooBig) => const_eval_invalid_meta,
            UnterminatedCString(_) => const_eval_unterminated_c_string,
            PointerUseAfterFree(_) => const_eval_pointer_use_after_free,
            PointerOutOfBounds { ptr_size: Size::ZERO, .. } => const_eval_zst_pointer_out_of_bounds,
            PointerOutOfBounds { .. } => const_eval_pointer_out_of_bounds,
            DanglingIntPointer(0, _) => const_eval_dangling_null_pointer,
            DanglingIntPointer(_, _) => const_eval_dangling_int_pointer,
            AlignmentCheckFailed { .. } => const_eval_alignment_check_failed,
            WriteToReadOnly(_) => const_eval_write_to_read_only,
            DerefFunctionPointer(_) => const_eval_deref_function_pointer,
            DerefVTablePointer(_) => const_eval_deref_vtable_pointer,
            InvalidBool(_) => const_eval_invalid_bool,
            InvalidChar(_) => const_eval_invalid_char,
            InvalidTag(_) => const_eval_invalid_tag,
            InvalidFunctionPointer(_) => const_eval_invalid_function_pointer,
            InvalidVTablePointer(_) => const_eval_invalid_vtable_pointer,
            InvalidStr(_) => const_eval_invalid_str,
            InvalidUninitBytes(None) => const_eval_invalid_uninit_bytes_unknown,
            InvalidUninitBytes(Some(_)) => const_eval_invalid_uninit_bytes,
            DeadLocal => const_eval_dead_local,
            ScalarSizeMismatch(_) => const_eval_scalar_size_mismatch,
            UninhabitedEnumVariantWritten => const_eval_uninhabited_enum_variant_written,
            Validation(e) => e.diagnostic_message(),
            Custom(x) => (x.msg)(),
        }
    }

    fn add_args<G: EmissionGuarantee>(
        self,
        handler: &Handler,
        builder: &mut DiagnosticBuilder<'_, G>,
    ) {
        use UndefinedBehaviorInfo::*;
        match self {
            Ub(_)
            | Unreachable
            | DivisionByZero
            | RemainderByZero
            | DivisionOverflow
            | RemainderOverflow
            | PointerArithOverflow
            | InvalidMeta(InvalidMetaKind::SliceTooBig)
            | InvalidMeta(InvalidMetaKind::TooBig)
            | InvalidUninitBytes(None)
            | DeadLocal
            | UninhabitedEnumVariantWritten => {}
            BoundsCheckFailed { len, index } => {
                builder.set_arg("len", len);
                builder.set_arg("index", index);
            }
            UnterminatedCString(ptr) | InvalidFunctionPointer(ptr) | InvalidVTablePointer(ptr) => {
                builder.set_arg("pointer", ptr);
            }
            PointerUseAfterFree(allocation) => {
                builder.set_arg("allocation", allocation);
            }
            PointerOutOfBounds { alloc_id, alloc_size, ptr_offset, ptr_size, msg } => {
                builder
                    .set_arg("alloc_id", alloc_id)
                    .set_arg("alloc_size", alloc_size.bytes())
                    .set_arg("ptr_offset", ptr_offset)
                    .set_arg("ptr_size", ptr_size.bytes())
                    .set_arg("bad_pointer_message", bad_pointer_message(msg, handler));
            }
            DanglingIntPointer(ptr, msg) => {
                if ptr != 0 {
                    builder.set_arg("pointer", format!("{ptr:#x}[noalloc]"));
                }

                builder.set_arg("bad_pointer_message", bad_pointer_message(msg, handler));
            }
            AlignmentCheckFailed { required, has } => {
                builder.set_arg("required", required.bytes());
                builder.set_arg("has", has.bytes());
            }
            WriteToReadOnly(alloc) | DerefFunctionPointer(alloc) | DerefVTablePointer(alloc) => {
                builder.set_arg("allocation", alloc);
            }
            InvalidBool(b) => {
                builder.set_arg("value", format!("{b:02x}"));
            }
            InvalidChar(c) => {
                builder.set_arg("value", format!("{c:08x}"));
            }
            InvalidTag(tag) => {
                builder.set_arg("tag", format!("{tag:x}"));
            }
            InvalidStr(err) => {
                builder.set_arg("err", format!("{err}"));
            }
            InvalidUninitBytes(Some((alloc, info))) => {
                builder.set_arg("alloc", alloc);
                builder.set_arg("access", info.access);
                builder.set_arg("uninit", info.uninit);
            }
            ScalarSizeMismatch(info) => {
                builder.set_arg("target_size", info.target_size);
                builder.set_arg("data_size", info.data_size);
            }
            Validation(e) => e.add_args(handler, builder),
            Custom(custom) => {
                (custom.add_args)(&mut |name, value| {
                    builder.set_arg(name, value);
                });
            }
        }
    }
}

impl<'tcx> ReportErrorExt for ValidationErrorInfo<'tcx> {
    fn diagnostic_message(&self) -> DiagnosticMessage {
        use crate::fluent_generated::*;
        use rustc_middle::mir::interpret::ValidationErrorKind::*;
        match self.kind {
            PtrToUninhabited { ptr_kind: PointerKind::Box, .. } => const_eval_box_to_uninhabited,
            PtrToUninhabited { ptr_kind: PointerKind::Ref, .. } => const_eval_ref_to_uninhabited,

            PtrToStatic { ptr_kind: PointerKind::Box } => const_eval_box_to_static,
            PtrToStatic { ptr_kind: PointerKind::Ref } => const_eval_ref_to_static,

            PtrToMut { ptr_kind: PointerKind::Box } => const_eval_box_to_mut,
            PtrToMut { ptr_kind: PointerKind::Ref } => const_eval_ref_to_mut,

            ExpectedNonPtr { .. } => const_eval_expected_non_ptr,
            MutableRefInConst => const_eval_mutable_ref_in_const,
            NullFnPtr => const_eval_null_fn_ptr,
            NeverVal => const_eval_never_val,
            NullablePtrOutOfRange { .. } => const_eval_nullable_ptr_out_of_range,
            PtrOutOfRange { .. } => const_eval_ptr_out_of_range,
            OutOfRange { .. } => const_eval_out_of_range,
            UnsafeCell => const_eval_unsafe_cell,
            UninhabitedVal { .. } => const_eval_uninhabited_val,
            InvalidEnumTag { .. } => const_eval_invalid_enum_tag,
            UninitEnumTag => const_eval_uninit_enum_tag,
            UninitStr => const_eval_uninit_str,
            Uninit { expected: ExpectedKind::Bool } => const_eval_uninit_bool,
            Uninit { expected: ExpectedKind::Reference } => const_eval_uninit_ref,
            Uninit { expected: ExpectedKind::Box } => const_eval_uninit_box,
            Uninit { expected: ExpectedKind::RawPtr } => const_eval_uninit_raw_ptr,
            Uninit { expected: ExpectedKind::InitScalar } => const_eval_uninit_init_scalar,
            Uninit { expected: ExpectedKind::Char } => const_eval_uninit_char,
            Uninit { expected: ExpectedKind::Float } => const_eval_uninit_float,
            Uninit { expected: ExpectedKind::Int } => const_eval_uninit_int,
            Uninit { expected: ExpectedKind::FnPtr } => const_eval_uninit_fn_ptr,
            UninitVal => const_eval_uninit,
            InvalidVTablePtr { .. } => const_eval_invalid_vtable_ptr,
            InvalidMetaSliceTooLarge { ptr_kind: PointerKind::Box } => {
                const_eval_invalid_box_slice_meta
            }
            InvalidMetaSliceTooLarge { ptr_kind: PointerKind::Ref } => {
                const_eval_invalid_ref_slice_meta
            }

            InvalidMetaTooLarge { ptr_kind: PointerKind::Box } => const_eval_invalid_box_meta,
            InvalidMetaTooLarge { ptr_kind: PointerKind::Ref } => const_eval_invalid_ref_meta,
            UnalignedPtr { ptr_kind: PointerKind::Ref, .. } => const_eval_unaligned_ref,
            UnalignedPtr { ptr_kind: PointerKind::Box, .. } => const_eval_unaligned_box,

            NullPtr { ptr_kind: PointerKind::Box } => const_eval_null_box,
            NullPtr { ptr_kind: PointerKind::Ref } => const_eval_null_ref,
            DanglingPtrNoProvenance { ptr_kind: PointerKind::Box, .. } => {
                const_eval_dangling_box_no_provenance
            }
            DanglingPtrNoProvenance { ptr_kind: PointerKind::Ref, .. } => {
                const_eval_dangling_ref_no_provenance
            }
            DanglingPtrOutOfBounds { ptr_kind: PointerKind::Box } => {
                const_eval_dangling_box_out_of_bounds
            }
            DanglingPtrOutOfBounds { ptr_kind: PointerKind::Ref } => {
                const_eval_dangling_ref_out_of_bounds
            }
            DanglingPtrUseAfterFree { ptr_kind: PointerKind::Box } => {
                const_eval_dangling_box_use_after_free
            }
            DanglingPtrUseAfterFree { ptr_kind: PointerKind::Ref } => {
                const_eval_dangling_ref_use_after_free
            }
            InvalidBool { .. } => const_eval_validation_invalid_bool,
            InvalidChar { .. } => const_eval_validation_invalid_char,
            InvalidFnPtr { .. } => const_eval_invalid_fn_ptr,
        }
    }

    fn add_args<G: EmissionGuarantee>(self, handler: &Handler, err: &mut DiagnosticBuilder<'_, G>) {
        use crate::fluent_generated as fluent;
        use rustc_middle::mir::interpret::ValidationErrorKind::*;

        let message = if let Some(path) = self.path {
            handler.eagerly_translate_to_string(
                fluent::const_eval_invalid_value_with_path,
                [("path".into(), DiagnosticArgValue::Str(path.into()))].iter().map(|(a, b)| (a, b)),
            )
        } else {
            handler.eagerly_translate_to_string(fluent::const_eval_invalid_value, [].into_iter())
        };

        err.set_arg("front_matter", message);

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

        match self.kind {
            PtrToUninhabited { ty, .. } | UninhabitedVal { ty } => {
                err.set_arg("ty", ty);
            }
            ExpectedNonPtr { value }
            | InvalidEnumTag { value }
            | InvalidVTablePtr { value }
            | InvalidBool { value }
            | InvalidChar { value }
            | InvalidFnPtr { value } => {
                err.set_arg("value", value);
            }
            NullablePtrOutOfRange { range, max_value } | PtrOutOfRange { range, max_value } => {
                add_range_arg(range, max_value, handler, err)
            }
            OutOfRange { range, max_value, value } => {
                err.set_arg("value", value);
                add_range_arg(range, max_value, handler, err);
            }
            UnalignedPtr { required_bytes, found_bytes, .. } => {
                err.set_arg("required_bytes", required_bytes);
                err.set_arg("found_bytes", found_bytes);
            }
            DanglingPtrNoProvenance { pointer, .. } => {
                err.set_arg("pointer", pointer);
            }
            NullPtr { .. }
            | PtrToStatic { .. }
            | PtrToMut { .. }
            | MutableRefInConst
            | NullFnPtr
            | NeverVal
            | UnsafeCell
            | UninitEnumTag
            | UninitStr
            | Uninit { .. }
            | UninitVal
            | InvalidMetaSliceTooLarge { .. }
            | InvalidMetaTooLarge { .. }
            | DanglingPtrUseAfterFree { .. }
            | DanglingPtrOutOfBounds { .. } => {}
        }
    }
}

impl ReportErrorExt for UnsupportedOpInfo {
    fn diagnostic_message(&self) -> DiagnosticMessage {
        use crate::fluent_generated::*;
        match self {
            UnsupportedOpInfo::Unsupported(s) => s.clone().into(),
            UnsupportedOpInfo::PartialPointerOverwrite(_) => const_eval_partial_pointer_overwrite,
            UnsupportedOpInfo::PartialPointerCopy(_) => const_eval_partial_pointer_copy,
            UnsupportedOpInfo::ReadPointerAsBytes => const_eval_read_pointer_as_bytes,
            UnsupportedOpInfo::ThreadLocalStatic(_) => const_eval_thread_local_static,
            UnsupportedOpInfo::ReadExternStatic(_) => const_eval_read_extern_static,
        }
    }
    fn add_args<G: EmissionGuarantee>(self, _: &Handler, builder: &mut DiagnosticBuilder<'_, G>) {
        use crate::fluent_generated::*;

        use UnsupportedOpInfo::*;
        if let ReadPointerAsBytes | PartialPointerOverwrite(_) | PartialPointerCopy(_) = self {
            builder.help(const_eval_ptr_as_bytes_1);
            builder.help(const_eval_ptr_as_bytes_2);
        }
        match self {
            Unsupported(_) | ReadPointerAsBytes => {}
            PartialPointerOverwrite(ptr) | PartialPointerCopy(ptr) => {
                builder.set_arg("ptr", ptr);
            }
            ThreadLocalStatic(did) | ReadExternStatic(did) => {
                builder.set_arg("did", format!("{did:?}"));
            }
        }
    }
}

impl<'tcx> ReportErrorExt for InterpError<'tcx> {
    fn diagnostic_message(&self) -> DiagnosticMessage {
        match self {
            InterpError::UndefinedBehavior(ub) => ub.diagnostic_message(),
            InterpError::Unsupported(e) => e.diagnostic_message(),
            InterpError::InvalidProgram(e) => e.diagnostic_message(),
            InterpError::ResourceExhaustion(e) => e.diagnostic_message(),
            InterpError::MachineStop(e) => e.diagnostic_message(),
        }
    }
    fn add_args<G: EmissionGuarantee>(
        self,
        handler: &Handler,
        builder: &mut DiagnosticBuilder<'_, G>,
    ) {
        match self {
            InterpError::UndefinedBehavior(ub) => ub.add_args(handler, builder),
            InterpError::Unsupported(e) => e.add_args(handler, builder),
            InterpError::InvalidProgram(e) => e.add_args(handler, builder),
            InterpError::ResourceExhaustion(e) => e.add_args(handler, builder),
            InterpError::MachineStop(e) => e.add_args(&mut |name, value| {
                builder.set_arg(name, value);
            }),
        }
    }
}

impl<'tcx> ReportErrorExt for InvalidProgramInfo<'tcx> {
    fn diagnostic_message(&self) -> DiagnosticMessage {
        use crate::fluent_generated::*;
        match self {
            InvalidProgramInfo::TooGeneric => const_eval_too_generic,
            InvalidProgramInfo::AlreadyReported(_) => const_eval_already_reported,
            InvalidProgramInfo::Layout(e) => e.diagnostic_message(),
            InvalidProgramInfo::FnAbiAdjustForForeignAbi(_) => {
                rustc_middle::error::middle_adjust_for_foreign_abi_error
            }
            InvalidProgramInfo::SizeOfUnsizedType(_) => const_eval_size_of_unsized,
            InvalidProgramInfo::UninitUnsizedLocal => const_eval_uninit_unsized_local,
        }
    }
    fn add_args<G: EmissionGuarantee>(
        self,
        handler: &Handler,
        builder: &mut DiagnosticBuilder<'_, G>,
    ) {
        match self {
            InvalidProgramInfo::TooGeneric
            | InvalidProgramInfo::AlreadyReported(_)
            | InvalidProgramInfo::UninitUnsizedLocal => {}
            InvalidProgramInfo::Layout(e) => {
                let diag: DiagnosticBuilder<'_, ()> = e.into_diagnostic().into_diagnostic(handler);
                for (name, val) in diag.args() {
                    builder.set_arg(name.clone(), val.clone());
                }
                diag.cancel();
            }
            InvalidProgramInfo::FnAbiAdjustForForeignAbi(
                AdjustForForeignAbiError::Unsupported { arch, abi },
            ) => {
                builder.set_arg("arch", arch);
                builder.set_arg("abi", abi.name());
            }
            InvalidProgramInfo::SizeOfUnsizedType(ty) => {
                builder.set_arg("ty", ty);
            }
        }
    }
}

impl ReportErrorExt for ResourceExhaustionInfo {
    fn diagnostic_message(&self) -> DiagnosticMessage {
        use crate::fluent_generated::*;
        match self {
            ResourceExhaustionInfo::StackFrameLimitReached => const_eval_stack_frame_limit_reached,
            ResourceExhaustionInfo::MemoryExhausted => const_eval_memory_exhausted,
            ResourceExhaustionInfo::AddressSpaceFull => const_eval_address_space_full,
        }
    }
    fn add_args<G: EmissionGuarantee>(self, _: &Handler, _: &mut DiagnosticBuilder<'_, G>) {}
}
