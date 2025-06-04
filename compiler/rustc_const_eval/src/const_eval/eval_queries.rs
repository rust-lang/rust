use std::sync::atomic::Ordering::Relaxed;

use either::{Left, Right};
use rustc_abi::{self as abi, BackendRepr};
use rustc_errors::E0080;
use rustc_hir::def::DefKind;
use rustc_middle::mir::interpret::{AllocId, ErrorHandled, InterpErrorInfo, ReportedErrorInfo};
use rustc_middle::mir::{self, ConstAlloc, ConstValue};
use rustc_middle::query::TyCtxtAt;
use rustc_middle::ty::layout::{HasTypingEnv, LayoutOf};
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_middle::{bug, throw_inval};
use rustc_span::def_id::LocalDefId;
use rustc_span::{DUMMY_SP, Span};
use tracing::{debug, instrument, trace};

use super::{CanAccessMutGlobal, CompileTimeInterpCx, CompileTimeMachine};
use crate::const_eval::CheckAlignment;
use crate::interpret::{
    CtfeValidationMode, GlobalId, Immediate, InternKind, InternResult, InterpCx, InterpErrorKind,
    InterpResult, MPlaceTy, MemoryKind, OpTy, RefTracking, StackPopCleanup, create_static_alloc,
    eval_nullary_intrinsic, intern_const_alloc_recursive, interp_ok, throw_exhaust,
};
use crate::{CTRL_C_RECEIVED, errors};

// Returns a pointer to where the result lives
#[instrument(level = "trace", skip(ecx, body))]
fn eval_body_using_ecx<'tcx, R: InterpretationResult<'tcx>>(
    ecx: &mut CompileTimeInterpCx<'tcx>,
    cid: GlobalId<'tcx>,
    body: &'tcx mir::Body<'tcx>,
) -> InterpResult<'tcx, R> {
    let tcx = *ecx.tcx;
    assert!(
        cid.promoted.is_some()
            || matches!(
                ecx.tcx.def_kind(cid.instance.def_id()),
                DefKind::Const
                    | DefKind::Static { .. }
                    | DefKind::ConstParam
                    | DefKind::AnonConst
                    | DefKind::InlineConst
                    | DefKind::AssocConst
            ),
        "Unexpected DefKind: {:?}",
        ecx.tcx.def_kind(cid.instance.def_id())
    );
    let layout = ecx.layout_of(body.bound_return_ty().instantiate(tcx, cid.instance.args))?;
    assert!(layout.is_sized());

    let intern_kind = if cid.promoted.is_some() {
        InternKind::Promoted
    } else {
        match tcx.static_mutability(cid.instance.def_id()) {
            Some(m) => InternKind::Static(m),
            None => InternKind::Constant,
        }
    };

    let ret = if let InternKind::Static(_) = intern_kind {
        create_static_alloc(ecx, cid.instance.def_id().expect_local(), layout)?
    } else {
        ecx.allocate(layout, MemoryKind::Stack)?
    };

    trace!(
        "eval_body_using_ecx: pushing stack frame for global: {}{}",
        with_no_trimmed_paths!(ecx.tcx.def_path_str(cid.instance.def_id())),
        cid.promoted.map_or_else(String::new, |p| format!("::{p:?}"))
    );

    // This can't use `init_stack_frame` since `body` is not a function,
    // so computing its ABI would fail. It's also not worth it since there are no arguments to pass.
    ecx.push_stack_frame_raw(
        cid.instance,
        body,
        &ret.clone().into(),
        StackPopCleanup::Root { cleanup: false },
    )?;
    ecx.storage_live_for_always_live_locals()?;

    // The main interpreter loop.
    while ecx.step()? {
        if CTRL_C_RECEIVED.load(Relaxed) {
            throw_exhaust!(Interrupted);
        }
    }

    // Intern the result
    let intern_result = intern_const_alloc_recursive(ecx, intern_kind, &ret);

    // Since evaluation had no errors, validate the resulting constant.
    const_validate_mplace(ecx, &ret, cid)?;

    // Only report this after validation, as validaiton produces much better diagnostics.
    // FIXME: ensure validation always reports this and stop making interning care about it.

    match intern_result {
        Ok(()) => {}
        Err(InternResult::FoundDanglingPointer) => {
            throw_inval!(AlreadyReported(ReportedErrorInfo::non_const_eval_error(
                ecx.tcx
                    .dcx()
                    .emit_err(errors::DanglingPtrInFinal { span: ecx.tcx.span, kind: intern_kind }),
            )));
        }
        Err(InternResult::FoundBadMutablePointer) => {
            throw_inval!(AlreadyReported(ReportedErrorInfo::non_const_eval_error(
                ecx.tcx
                    .dcx()
                    .emit_err(errors::MutablePtrInFinal { span: ecx.tcx.span, kind: intern_kind }),
            )));
        }
    }

    interp_ok(R::make_result(ret, ecx))
}

/// The `InterpCx` is only meant to be used to do field and index projections into constants for
/// `simd_shuffle` and const patterns in match arms.
///
/// This should *not* be used to do any actual interpretation. In particular, alignment checks are
/// turned off!
///
/// The function containing the `match` that is currently being analyzed may have generic bounds
/// that inform us about the generic bounds of the constant. E.g., using an associated constant
/// of a function's generic parameter will require knowledge about the bounds on the generic
/// parameter. These bounds are passed to `mk_eval_cx` via the `ParamEnv` argument.
pub(crate) fn mk_eval_cx_to_read_const_val<'tcx>(
    tcx: TyCtxt<'tcx>,
    root_span: Span,
    typing_env: ty::TypingEnv<'tcx>,
    can_access_mut_global: CanAccessMutGlobal,
) -> CompileTimeInterpCx<'tcx> {
    debug!("mk_eval_cx: {:?}", typing_env);
    InterpCx::new(
        tcx,
        root_span,
        typing_env,
        CompileTimeMachine::new(can_access_mut_global, CheckAlignment::No),
    )
}

/// Create an interpreter context to inspect the given `ConstValue`.
/// Returns both the context and an `OpTy` that represents the constant.
pub fn mk_eval_cx_for_const_val<'tcx>(
    tcx: TyCtxtAt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    val: mir::ConstValue<'tcx>,
    ty: Ty<'tcx>,
) -> Option<(CompileTimeInterpCx<'tcx>, OpTy<'tcx>)> {
    let ecx = mk_eval_cx_to_read_const_val(tcx.tcx, tcx.span, typing_env, CanAccessMutGlobal::No);
    // FIXME: is it a problem to discard the error here?
    let op = ecx.const_val_to_op(val, ty, None).discard_err()?;
    Some((ecx, op))
}

/// This function converts an interpreter value into a MIR constant.
///
/// The `for_diagnostics` flag turns the usual rules for returning `ConstValue::Scalar` into a
/// best-effort attempt. This is not okay for use in const-eval sine it breaks invariants rustc
/// relies on, but it is okay for diagnostics which will just give up gracefully when they
/// encounter an `Indirect` they cannot handle.
#[instrument(skip(ecx), level = "debug")]
pub(super) fn op_to_const<'tcx>(
    ecx: &CompileTimeInterpCx<'tcx>,
    op: &OpTy<'tcx>,
    for_diagnostics: bool,
) -> ConstValue<'tcx> {
    // Handle ZST consistently and early.
    if op.layout.is_zst() {
        return ConstValue::ZeroSized;
    }

    // All scalar types should be stored as `ConstValue::Scalar`. This is needed to make
    // `ConstValue::try_to_scalar` efficient; we want that to work for *all* constants of scalar
    // type (it's used throughout the compiler and having it work just on literals is not enough)
    // and we want it to be fast (i.e., don't go to an `Allocation` and reconstruct the `Scalar`
    // from its byte-serialized form).
    let force_as_immediate = match op.layout.backend_repr {
        BackendRepr::Scalar(abi::Scalar::Initialized { .. }) => true,
        // We don't *force* `ConstValue::Slice` for `ScalarPair`. This has the advantage that if the
        // input `op` is a place, then turning it into a `ConstValue` and back into a `OpTy` will
        // not have to generate any duplicate allocations (we preserve the original `AllocId` in
        // `ConstValue::Indirect`). It means accessing the contents of a slice can be slow (since
        // they can be stored as `ConstValue::Indirect`), but that's not relevant since we barely
        // ever have to do this. (`try_get_slice_bytes_for_diagnostics` exists to provide this
        // functionality.)
        _ => false,
    };
    let immediate = if force_as_immediate {
        match ecx.read_immediate(op).report_err() {
            Ok(imm) => Right(imm),
            Err(err) => {
                if for_diagnostics {
                    // This discard the error, but for diagnostics that's okay.
                    op.as_mplace_or_imm()
                } else {
                    panic!("normalization works on validated constants: {err:?}")
                }
            }
        }
    } else {
        op.as_mplace_or_imm()
    };

    debug!(?immediate);

    match immediate {
        Left(ref mplace) => {
            // We know `offset` is relative to the allocation, so we can use `into_parts`.
            let (prov, offset) = mplace.ptr().into_parts();
            let alloc_id = prov.expect("cannot have `fake` place for non-ZST type").alloc_id();
            ConstValue::Indirect { alloc_id, offset }
        }
        // see comment on `let force_as_immediate` above
        Right(imm) => match *imm {
            Immediate::Scalar(x) => ConstValue::Scalar(x),
            Immediate::ScalarPair(a, b) => {
                debug!("ScalarPair(a: {:?}, b: {:?})", a, b);
                // This codepath solely exists for `valtree_to_const_value` to not need to generate
                // a `ConstValue::Indirect` for wide references, so it is tightly restricted to just
                // that case.
                let pointee_ty = imm.layout.ty.builtin_deref(false).unwrap(); // `false` = no raw ptrs
                debug_assert!(
                    matches!(
                        ecx.tcx.struct_tail_for_codegen(pointee_ty, ecx.typing_env()).kind(),
                        ty::Str | ty::Slice(..),
                    ),
                    "`ConstValue::Slice` is for slice-tailed types only, but got {}",
                    imm.layout.ty,
                );
                let msg = "`op_to_const` on an immediate scalar pair must only be used on slice references to the beginning of an actual allocation";
                // We know `offset` is relative to the allocation, so we can use `into_parts`.
                let (prov, offset) = a.to_pointer(ecx).expect(msg).into_parts();
                let alloc_id = prov.expect(msg).alloc_id();
                let data = ecx.tcx.global_alloc(alloc_id).unwrap_memory();
                assert!(offset == abi::Size::ZERO, "{}", msg);
                let meta = b.to_target_usize(ecx).expect(msg);
                ConstValue::Slice { data, meta }
            }
            Immediate::Uninit => bug!("`Uninit` is not a valid value for {}", op.layout.ty),
        },
    }
}

#[instrument(skip(tcx), level = "debug", ret)]
pub(crate) fn turn_into_const_value<'tcx>(
    tcx: TyCtxt<'tcx>,
    constant: ConstAlloc<'tcx>,
    key: ty::PseudoCanonicalInput<'tcx, GlobalId<'tcx>>,
) -> ConstValue<'tcx> {
    let cid = key.value;
    let def_id = cid.instance.def.def_id();
    let is_static = tcx.is_static(def_id);
    // This is just accessing an already computed constant, so no need to check alignment here.
    let ecx = mk_eval_cx_to_read_const_val(
        tcx,
        tcx.def_span(key.value.instance.def_id()),
        key.typing_env,
        CanAccessMutGlobal::from(is_static),
    );

    let mplace = ecx.raw_const_to_mplace(constant).expect(
        "can only fail if layout computation failed, \
        which should have given a good error before ever invoking this function",
    );
    assert!(
        !is_static || cid.promoted.is_some(),
        "the `eval_to_const_value_raw` query should not be used for statics, use `eval_to_allocation` instead"
    );

    // Turn this into a proper constant.
    op_to_const(&ecx, &mplace.into(), /* for diagnostics */ false)
}

#[instrument(skip(tcx), level = "debug")]
pub fn eval_to_const_value_raw_provider<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: ty::PseudoCanonicalInput<'tcx, GlobalId<'tcx>>,
) -> ::rustc_middle::mir::interpret::EvalToConstValueResult<'tcx> {
    // We call `const_eval` for zero arg intrinsics, too, in order to cache their value.
    // Catch such calls and evaluate them instead of trying to load a constant's MIR.
    if let ty::InstanceKind::Intrinsic(def_id) = key.value.instance.def {
        let ty = key.value.instance.ty(tcx, key.typing_env);
        let ty::FnDef(_, args) = ty.kind() else {
            bug!("intrinsic with type {:?}", ty);
        };
        return eval_nullary_intrinsic(tcx, key.typing_env, def_id, args).report_err().map_err(
            |error| {
                let span = tcx.def_span(def_id);

                // FIXME(oli-obk): why don't we have any tests for this code path?
                super::report(
                    tcx,
                    error.into_kind(),
                    span,
                    || (span, vec![]),
                    |diag, span, _| {
                        diag.span_label(
                            span,
                            crate::fluent_generated::const_eval_nullary_intrinsic_fail,
                        );
                    },
                )
            },
        );
    }

    tcx.eval_to_allocation_raw(key).map(|val| turn_into_const_value(tcx, val, key))
}

#[instrument(skip(tcx), level = "debug")]
pub fn eval_static_initializer_provider<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: LocalDefId,
) -> ::rustc_middle::mir::interpret::EvalStaticInitializerRawResult<'tcx> {
    assert!(tcx.is_static(def_id.to_def_id()));

    let instance = ty::Instance::mono(tcx, def_id.to_def_id());
    let cid = rustc_middle::mir::interpret::GlobalId { instance, promoted: None };
    eval_in_interpreter(tcx, cid, ty::TypingEnv::fully_monomorphized())
}

pub trait InterpretationResult<'tcx> {
    /// This function takes the place where the result of the evaluation is stored
    /// and prepares it for returning it in the appropriate format needed by the specific
    /// evaluation query.
    fn make_result(
        mplace: MPlaceTy<'tcx>,
        ecx: &mut InterpCx<'tcx, CompileTimeMachine<'tcx>>,
    ) -> Self;
}

impl<'tcx> InterpretationResult<'tcx> for ConstAlloc<'tcx> {
    fn make_result(
        mplace: MPlaceTy<'tcx>,
        _ecx: &mut InterpCx<'tcx, CompileTimeMachine<'tcx>>,
    ) -> Self {
        ConstAlloc { alloc_id: mplace.ptr().provenance.unwrap().alloc_id(), ty: mplace.layout.ty }
    }
}

#[instrument(skip(tcx), level = "debug")]
pub fn eval_to_allocation_raw_provider<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: ty::PseudoCanonicalInput<'tcx, GlobalId<'tcx>>,
) -> ::rustc_middle::mir::interpret::EvalToAllocationRawResult<'tcx> {
    // This shouldn't be used for statics, since statics are conceptually places,
    // not values -- so what we do here could break pointer identity.
    assert!(key.value.promoted.is_some() || !tcx.is_static(key.value.instance.def_id()));
    // Const eval always happens in PostAnalysis mode . See the comment in
    // `InterpCx::new` for more details.
    debug_assert_eq!(key.typing_env.typing_mode, ty::TypingMode::PostAnalysis);
    if cfg!(debug_assertions) {
        // Make sure we format the instance even if we do not print it.
        // This serves as a regression test against an ICE on printing.
        // The next two lines concatenated contain some discussion:
        // https://rust-lang.zulipchat.com/#narrow/stream/146212-t-compiler.2Fconst-eval/
        // subject/anon_const_instance_printing/near/135980032
        let instance = with_no_trimmed_paths!(key.value.instance.to_string());
        trace!("const eval: {:?} ({})", key, instance);
    }

    eval_in_interpreter(tcx, key.value, key.typing_env)
}

fn eval_in_interpreter<'tcx, R: InterpretationResult<'tcx>>(
    tcx: TyCtxt<'tcx>,
    cid: GlobalId<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
) -> Result<R, ErrorHandled> {
    let def = cid.instance.def.def_id();
    let is_static = tcx.is_static(def);

    let mut ecx = InterpCx::new(
        tcx,
        tcx.def_span(def),
        typing_env,
        // Statics (and promoteds inside statics) may access mutable global memory, because unlike consts
        // they do not have to behave "as if" they were evaluated at runtime.
        // For consts however we want to ensure they behave "as if" they were evaluated at runtime,
        // so we have to reject reading mutable global memory.
        CompileTimeMachine::new(CanAccessMutGlobal::from(is_static), CheckAlignment::Error),
    );
    let res = ecx.load_mir(cid.instance.def, cid.promoted);
    res.and_then(|body| eval_body_using_ecx(&mut ecx, cid, body))
        .report_err()
        .map_err(|error| report_eval_error(&ecx, cid, error))
}

#[inline(always)]
fn const_validate_mplace<'tcx>(
    ecx: &mut InterpCx<'tcx, CompileTimeMachine<'tcx>>,
    mplace: &MPlaceTy<'tcx>,
    cid: GlobalId<'tcx>,
) -> Result<(), ErrorHandled> {
    let alloc_id = mplace.ptr().provenance.unwrap().alloc_id();
    let mut ref_tracking = RefTracking::new(mplace.clone());
    let mut inner = false;
    while let Some((mplace, path)) = ref_tracking.next() {
        let mode = match ecx.tcx.static_mutability(cid.instance.def_id()) {
            _ if cid.promoted.is_some() => CtfeValidationMode::Promoted,
            Some(mutbl) => CtfeValidationMode::Static { mutbl }, // a `static`
            None => {
                // This is a normal `const` (not promoted).
                // The outermost allocation is always only copied, so having `UnsafeCell` in there
                // is okay despite them being in immutable memory.
                CtfeValidationMode::Const { allow_immutable_unsafe_cell: !inner }
            }
        };
        ecx.const_validate_operand(&mplace.into(), path, &mut ref_tracking, mode)
            .report_err()
            // Instead of just reporting the `InterpError` via the usual machinery, we give a more targeted
            // error about the validation failure.
            .map_err(|error| report_validation_error(&ecx, cid, error, alloc_id))?;
        inner = true;
    }

    Ok(())
}

#[inline(never)]
fn report_eval_error<'tcx>(
    ecx: &InterpCx<'tcx, CompileTimeMachine<'tcx>>,
    cid: GlobalId<'tcx>,
    error: InterpErrorInfo<'tcx>,
) -> ErrorHandled {
    let (error, backtrace) = error.into_parts();
    backtrace.print_backtrace();

    let (kind, instance) = if ecx.tcx.is_static(cid.instance.def_id()) {
        ("static", String::new())
    } else {
        // If the current item has generics, we'd like to enrich the message with the
        // instance and its args: to show the actual compile-time values, in addition to
        // the expression, leading to the const eval error.
        let instance = &cid.instance;
        if !instance.args.is_empty() {
            let instance = with_no_trimmed_paths!(instance.to_string());
            ("const_with_path", instance)
        } else {
            ("const", String::new())
        }
    };

    super::report(
        *ecx.tcx,
        error,
        DUMMY_SP,
        || super::get_span_and_frames(ecx.tcx, ecx.stack()),
        |diag, span, frames| {
            // FIXME(oli-obk): figure out how to use structured diagnostics again.
            diag.code(E0080);
            diag.span_label(span, crate::fluent_generated::const_eval_error);
            diag.arg("instance", instance);
            diag.arg("error_kind", kind);
            for frame in frames {
                diag.subdiagnostic(frame);
            }
        },
    )
}

#[inline(never)]
fn report_validation_error<'tcx>(
    ecx: &InterpCx<'tcx, CompileTimeMachine<'tcx>>,
    cid: GlobalId<'tcx>,
    error: InterpErrorInfo<'tcx>,
    alloc_id: AllocId,
) -> ErrorHandled {
    if !matches!(error.kind(), InterpErrorKind::UndefinedBehavior(_)) {
        // Some other error happened during validation, e.g. an unsupported operation.
        return report_eval_error(ecx, cid, error);
    }

    let (error, backtrace) = error.into_parts();
    backtrace.print_backtrace();

    let bytes = ecx.print_alloc_bytes_for_diagnostics(alloc_id);
    let info = ecx.get_alloc_info(alloc_id);
    let raw_bytes =
        errors::RawBytesNote { size: info.size.bytes(), align: info.align.bytes(), bytes };

    crate::const_eval::report(
        *ecx.tcx,
        error,
        DUMMY_SP,
        || crate::const_eval::get_span_and_frames(ecx.tcx, ecx.stack()),
        move |diag, span, frames| {
            // FIXME(oli-obk): figure out how to use structured diagnostics again.
            diag.code(E0080);
            diag.span_label(span, crate::fluent_generated::const_eval_validation_failure);
            diag.note(crate::fluent_generated::const_eval_validation_failure_note);
            for frame in frames {
                diag.subdiagnostic(frame);
            }
            diag.subdiagnostic(raw_bytes);
        },
    )
}
