use std::mem;

use rustc_errors::{
    DiagnosticArgValue, DiagnosticBuilder, DiagnosticMessage, EmissionGuarantee, Handler,
    IntoDiagnostic, IntoDiagnosticArg,
};
use rustc_middle::mir::interpret::{
    CheckInAllocMsg, ExpectedKind, InvalidMetaKind, InvalidProgramInfo, PointerKind,
    ResourceExhaustionInfo, UndefinedBehaviorInfo, UnsupportedOpInfo, ValidationErrorInfo,
    ValidationErrorKind,
};
use rustc_middle::mir::AssertKind;
use rustc_middle::ty::TyCtxt;
use rustc_middle::ty::{layout::LayoutError, ConstInt};
use rustc_span::source_map::Spanned;
use rustc_span::{ErrorGuaranteed, Span, Symbol};
use rustc_target::abi::call::AdjustForForeignAbiError;
use rustc_target::abi::{Size, WrappingRange};

use super::InterpCx;
use crate::errors::{self, FrameNote};
use crate::interpret::{ErrorHandled, InterpError, InterpErrorInfo, Machine, MachineStopType};

/// The CTFE machine has some custom error kinds.
#[derive(Clone, Debug)]
pub enum ConstEvalErrKind {
    ConstAccessesStatic,
    ModifiedGlobal,
    AssertFailure(AssertKind<ConstInt>),
    Panic { msg: Symbol, line: u32, col: u32, file: Symbol },
    Abort(String),
}

impl MachineStopType for ConstEvalErrKind {
    fn diagnostic_message(&self) -> DiagnosticMessage {
        use crate::fluent_generated::*;
        use ConstEvalErrKind::*;
        match self {
            ConstAccessesStatic => const_eval_const_accesses_static,
            ModifiedGlobal => const_eval_modified_global,
            Panic { .. } => const_eval_panic,
            AssertFailure(x) => x.diagnostic_message(),
            Abort(msg) => msg.to_string().into(),
        }
    }
    fn add_args(
        self: Box<Self>,
        adder: &mut dyn FnMut(std::borrow::Cow<'static, str>, DiagnosticArgValue<'static>),
    ) {
        use ConstEvalErrKind::*;
        match *self {
            ConstAccessesStatic | ModifiedGlobal | Abort(_) => {}
            AssertFailure(kind) => kind.add_args(adder),
            Panic { msg, line, col, file } => {
                adder("msg".into(), msg.into_diagnostic_arg());
                adder("file".into(), file.into_diagnostic_arg());
                adder("line".into(), line.into_diagnostic_arg());
                adder("col".into(), col.into_diagnostic_arg());
            }
        }
    }
}

// The errors become `MachineStop` with plain strings when being raised.
// `ConstEvalErr` (in `librustc_middle/mir/interpret/error.rs`) knows to
// handle these.
impl<'tcx> Into<InterpErrorInfo<'tcx>> for ConstEvalErrKind {
    fn into(self) -> InterpErrorInfo<'tcx> {
        err_machine_stop!(self).into()
    }
}

pub fn get_span_and_frames<'tcx, 'mir, M: Machine<'mir, 'tcx>>(
    ecx: &InterpCx<'mir, 'tcx, M>,
) -> (Span, Vec<errors::FrameNote>)
where
    'tcx: 'mir,
{
    let mut stacktrace = ecx.generate_stacktrace();
    // Filter out `requires_caller_location` frames.
    stacktrace.retain(|frame| !frame.instance.def.requires_caller_location(*ecx.tcx));
    let span = stacktrace.first().map(|f| f.span).unwrap_or(ecx.tcx.span);

    let mut frames = Vec::new();

    // Add notes to the backtrace. Don't print a single-line backtrace though.
    if stacktrace.len() > 1 {
        // Helper closure to print duplicated lines.
        let mut add_frame = |mut frame: errors::FrameNote| {
            frames.push(errors::FrameNote { times: 0, ..frame.clone() });
            // Don't print [... additional calls ...] if the number of lines is small
            if frame.times < 3 {
                let times = frame.times;
                frame.times = 0;
                frames.extend(std::iter::repeat(frame).take(times as usize));
            } else {
                frames.push(frame);
            }
        };

        let mut last_frame: Option<errors::FrameNote> = None;
        for frame_info in &stacktrace {
            let frame = frame_info.as_note(*ecx.tcx);
            match last_frame.as_mut() {
                Some(last_frame)
                    if last_frame.span == frame.span
                        && last_frame.where_ == frame.where_
                        && last_frame.instance == frame.instance =>
                {
                    last_frame.times += 1;
                }
                Some(last_frame) => {
                    add_frame(mem::replace(last_frame, frame));
                }
                None => {
                    last_frame = Some(frame);
                }
            }
        }
        if let Some(frame) = last_frame {
            add_frame(frame);
        }
    }

    (span, frames)
}

pub trait ReportErrorExt {
    /// Returns the diagnostic message for this error.
    fn diagnostic_message(&self) -> DiagnosticMessage;
    fn add_args<G: EmissionGuarantee>(
        self,
        handler: &Handler,
        builder: &mut DiagnosticBuilder<'_, G>,
    );
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
            Ub(msg) => (&**msg).into(),
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
        use ValidationErrorKind::*;
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
        use ValidationErrorKind::*;

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
            ResourceExhaustionInfo::StepLimitReached => const_eval_step_limit_reached,
            ResourceExhaustionInfo::MemoryExhausted => const_eval_memory_exhausted,
            ResourceExhaustionInfo::AddressSpaceFull => const_eval_address_space_full,
        }
    }
    fn add_args<G: EmissionGuarantee>(self, _: &Handler, _: &mut DiagnosticBuilder<'_, G>) {}
}

/// Create a diagnostic for a const eval error.
///
/// This will use the `mk` function for creating the error which will get passed labels according to
/// the `InterpError` and the span and a stacktrace of current execution according to
/// `get_span_and_frames`.
pub(super) fn report<'tcx, C, F, E>(
    tcx: TyCtxt<'tcx>,
    error: InterpError<'tcx>,
    span: Option<Span>,
    get_span_and_frames: C,
    mk: F,
) -> ErrorHandled
where
    C: FnOnce() -> (Span, Vec<FrameNote>),
    F: FnOnce(Span, Vec<FrameNote>) -> E,
    E: IntoDiagnostic<'tcx, ErrorGuaranteed>,
{
    // Special handling for certain errors
    match error {
        // Don't emit a new diagnostic for these errors
        err_inval!(Layout(LayoutError::Unknown(_))) | err_inval!(TooGeneric) => {
            ErrorHandled::TooGeneric
        }
        err_inval!(AlreadyReported(error_reported)) => ErrorHandled::Reported(error_reported),
        err_inval!(Layout(layout_error @ LayoutError::SizeOverflow(_))) => {
            // We must *always* hard error on these, even if the caller wants just a lint.
            // The `message` makes little sense here, this is a more serious error than the
            // caller thinks anyway.
            // See <https://github.com/rust-lang/rust/pull/63152>.
            let (our_span, frames) = get_span_and_frames();
            let span = span.unwrap_or(our_span);
            let mut err =
                tcx.sess.create_err(Spanned { span, node: layout_error.into_diagnostic() });
            err.code(rustc_errors::error_code!(E0080));
            let Some((mut err, handler)) = err.into_diagnostic() else {
                    panic!("did not emit diag");
                };
            for frame in frames {
                err.eager_subdiagnostic(handler, frame);
            }

            ErrorHandled::Reported(handler.emit_diagnostic(&mut err).unwrap().into())
        }
        _ => {
            // Report as hard error.
            let (our_span, frames) = get_span_and_frames();
            let span = span.unwrap_or(our_span);
            let err = mk(span, frames);
            let mut err = tcx.sess.create_err(err);

            let msg = error.diagnostic_message();
            error.add_args(&tcx.sess.parse_sess.span_diagnostic, &mut err);

            // Use *our* span to label the interp error
            err.span_label(our_span, msg);
            ErrorHandled::Reported(err.emit().into())
        }
    }
}
