use crate::const_eval::CheckAlignment;
use crate::errors::ConstEvalError;

use either::{Left, Right};

use rustc_hir::def::DefKind;
use rustc_middle::mir;
use rustc_middle::mir::interpret::{ErrorHandled, InterpErrorInfo};
use rustc_middle::mir::pretty::write_allocation_bytes;
use rustc_middle::traits::Reveal;
use rustc_middle::ty::layout::LayoutOf;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::source_map::Span;
use rustc_target::abi::{self, Abi};

use super::{CanAccessStatics, CompileTimeEvalContext, CompileTimeInterpreter};
use crate::errors;
use crate::interpret::eval_nullary_intrinsic;
use crate::interpret::{
    intern_const_alloc_recursive, Allocation, ConstAlloc, ConstValue, CtfeValidationMode, GlobalId,
    Immediate, InternKind, InterpCx, InterpError, InterpResult, MPlaceTy, MemoryKind, OpTy,
    RefTracking, StackPopCleanup,
};

// Returns a pointer to where the result lives
fn eval_body_using_ecx<'mir, 'tcx>(
    ecx: &mut CompileTimeEvalContext<'mir, 'tcx>,
    cid: GlobalId<'tcx>,
    body: &'mir mir::Body<'tcx>,
) -> InterpResult<'tcx, MPlaceTy<'tcx>> {
    debug!("eval_body_using_ecx: {:?}, {:?}", cid, ecx.param_env);
    let tcx = *ecx.tcx;
    assert!(
        cid.promoted.is_some()
            || matches!(
                ecx.tcx.def_kind(cid.instance.def_id()),
                DefKind::Const
                    | DefKind::Static(_)
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
    let ret = ecx.allocate(layout, MemoryKind::Stack)?;

    trace!(
        "eval_body_using_ecx: pushing stack frame for global: {}{}",
        with_no_trimmed_paths!(ecx.tcx.def_path_str(cid.instance.def_id())),
        cid.promoted.map_or_else(String::new, |p| format!("::promoted[{:?}]", p))
    );

    ecx.push_stack_frame(
        cid.instance,
        body,
        &ret.into(),
        StackPopCleanup::Root { cleanup: false },
    )?;

    // The main interpreter loop.
    while ecx.step()? {}

    // Intern the result
    let intern_kind = if cid.promoted.is_some() {
        InternKind::Promoted
    } else {
        match tcx.static_mutability(cid.instance.def_id()) {
            Some(m) => InternKind::Static(m),
            None => InternKind::Constant,
        }
    };
    ecx.machine.check_alignment = CheckAlignment::No; // interning doesn't need to respect alignment
    intern_const_alloc_recursive(ecx, intern_kind, &ret)?;
    // we leave alignment checks off, since this `ecx` will not be used for further evaluation anyway

    debug!("eval_body_using_ecx done: {:?}", *ret);
    Ok(ret)
}

/// The `InterpCx` is only meant to be used to do field and index projections into constants for
/// `simd_shuffle` and const patterns in match arms. It never performs alignment checks.
///
/// The function containing the `match` that is currently being analyzed may have generic bounds
/// that inform us about the generic bounds of the constant. E.g., using an associated constant
/// of a function's generic parameter will require knowledge about the bounds on the generic
/// parameter. These bounds are passed to `mk_eval_cx` via the `ParamEnv` argument.
pub(super) fn mk_eval_cx<'mir, 'tcx>(
    tcx: TyCtxt<'tcx>,
    root_span: Span,
    param_env: ty::ParamEnv<'tcx>,
    can_access_statics: CanAccessStatics,
) -> CompileTimeEvalContext<'mir, 'tcx> {
    debug!("mk_eval_cx: {:?}", param_env);
    InterpCx::new(
        tcx,
        root_span,
        param_env,
        CompileTimeInterpreter::new(can_access_statics, CheckAlignment::No),
    )
}

/// This function converts an interpreter value into a constant that is meant for use in the
/// type system.
#[instrument(skip(ecx), level = "debug")]
pub(super) fn op_to_const<'tcx>(
    ecx: &CompileTimeEvalContext<'_, 'tcx>,
    op: &OpTy<'tcx>,
) -> ConstValue<'tcx> {
    // We do not have value optimizations for everything.
    // Only scalars and slices, since they are very common.
    // Note that further down we turn scalars of uninitialized bits back to `ByRef`. These can result
    // from scalar unions that are initialized with one of their zero sized variants. We could
    // instead allow `ConstValue::Scalar` to store `ScalarMaybeUninit`, but that would affect all
    // the usual cases of extracting e.g. a `usize`, without there being a real use case for the
    // `Undef` situation.
    let try_as_immediate = match op.layout.abi {
        Abi::Scalar(abi::Scalar::Initialized { .. }) => true,
        Abi::ScalarPair(..) => match op.layout.ty.kind() {
            ty::Ref(_, inner, _) => match *inner.kind() {
                ty::Slice(elem) => elem == ecx.tcx.types.u8,
                ty::Str => true,
                _ => false,
            },
            _ => false,
        },
        _ => false,
    };
    let immediate = if try_as_immediate {
        Right(ecx.read_immediate(op).expect("normalization works on validated constants"))
    } else {
        // It is guaranteed that any non-slice scalar pair is actually ByRef here.
        // When we come back from raw const eval, we are always by-ref. The only way our op here is
        // by-val is if we are in destructure_mir_constant, i.e., if this is (a field of) something that we
        // "tried to make immediate" before. We wouldn't do that for non-slice scalar pairs or
        // structs containing such.
        op.as_mplace_or_imm()
    };

    debug!(?immediate);

    // We know `offset` is relative to the allocation, so we can use `into_parts`.
    let to_const_value = |mplace: &MPlaceTy<'_>| {
        debug!("to_const_value(mplace: {:?})", mplace);
        match mplace.ptr.into_parts() {
            (Some(alloc_id), offset) => {
                let alloc = ecx.tcx.global_alloc(alloc_id).unwrap_memory();
                ConstValue::ByRef { alloc, offset }
            }
            (None, offset) => {
                assert!(mplace.layout.is_zst());
                assert_eq!(
                    offset.bytes() % mplace.layout.align.abi.bytes(),
                    0,
                    "this MPlaceTy must come from a validated constant, thus we can assume the \
                alignment is correct",
                );
                ConstValue::ZeroSized
            }
        }
    };
    match immediate {
        Left(ref mplace) => to_const_value(mplace),
        // see comment on `let try_as_immediate` above
        Right(imm) => match *imm {
            _ if imm.layout.is_zst() => ConstValue::ZeroSized,
            Immediate::Scalar(x) => ConstValue::Scalar(x),
            Immediate::ScalarPair(a, b) => {
                debug!("ScalarPair(a: {:?}, b: {:?})", a, b);
                // We know `offset` is relative to the allocation, so we can use `into_parts`.
                let (data, start) = match a.to_pointer(ecx).unwrap().into_parts() {
                    (Some(alloc_id), offset) => {
                        (ecx.tcx.global_alloc(alloc_id).unwrap_memory(), offset.bytes())
                    }
                    (None, _offset) => (
                        ecx.tcx.mk_const_alloc(Allocation::from_bytes_byte_aligned_immutable(
                            b"" as &[u8],
                        )),
                        0,
                    ),
                };
                let len = b.to_target_usize(ecx).unwrap();
                let start = start.try_into().unwrap();
                let len: usize = len.try_into().unwrap();
                ConstValue::Slice { data, start, end: start + len }
            }
            Immediate::Uninit => to_const_value(&op.assert_mem_place()),
        },
    }
}

#[instrument(skip(tcx), level = "debug", ret)]
pub(crate) fn turn_into_const_value<'tcx>(
    tcx: TyCtxt<'tcx>,
    constant: ConstAlloc<'tcx>,
    key: ty::ParamEnvAnd<'tcx, GlobalId<'tcx>>,
) -> ConstValue<'tcx> {
    let cid = key.value;
    let def_id = cid.instance.def.def_id();
    let is_static = tcx.is_static(def_id);
    // This is just accessing an already computed constant, so no need to check alignment here.
    let ecx = mk_eval_cx(
        tcx,
        tcx.def_span(key.value.instance.def_id()),
        key.param_env,
        CanAccessStatics::from(is_static),
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
    op_to_const(&ecx, &mplace.into())
}

#[instrument(skip(tcx), level = "debug")]
pub fn eval_to_const_value_raw_provider<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: ty::ParamEnvAnd<'tcx, GlobalId<'tcx>>,
) -> ::rustc_middle::mir::interpret::EvalToConstValueResult<'tcx> {
    assert!(key.param_env.is_const());
    // see comment in eval_to_allocation_raw_provider for what we're doing here
    if key.param_env.reveal() == Reveal::All {
        let mut key = key;
        key.param_env = key.param_env.with_user_facing();
        match tcx.eval_to_const_value_raw(key) {
            // try again with reveal all as requested
            Err(ErrorHandled::TooGeneric) => {}
            // deduplicate calls
            other => return other,
        }
    }

    // We call `const_eval` for zero arg intrinsics, too, in order to cache their value.
    // Catch such calls and evaluate them instead of trying to load a constant's MIR.
    if let ty::InstanceDef::Intrinsic(def_id) = key.value.instance.def {
        let ty = key.value.instance.ty(tcx, key.param_env);
        let ty::FnDef(_, args) = ty.kind() else {
            bug!("intrinsic with type {:?}", ty);
        };
        return eval_nullary_intrinsic(tcx, key.param_env, def_id, args).map_err(|error| {
            let span = tcx.def_span(def_id);

            super::report(
                tcx,
                error.into_kind(),
                Some(span),
                || (span, vec![]),
                |span, _| errors::NullaryIntrinsicError { span },
            )
        });
    }

    tcx.eval_to_allocation_raw(key).map(|val| turn_into_const_value(tcx, val, key))
}

#[instrument(skip(tcx), level = "debug")]
pub fn eval_to_allocation_raw_provider<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: ty::ParamEnvAnd<'tcx, GlobalId<'tcx>>,
) -> ::rustc_middle::mir::interpret::EvalToAllocationRawResult<'tcx> {
    assert!(key.param_env.is_const());
    // Because the constant is computed twice (once per value of `Reveal`), we are at risk of
    // reporting the same error twice here. To resolve this, we check whether we can evaluate the
    // constant in the more restrictive `Reveal::UserFacing`, which most likely already was
    // computed. For a large percentage of constants that will already have succeeded. Only
    // associated constants of generic functions will fail due to not enough monomorphization
    // information being available.

    // In case we fail in the `UserFacing` variant, we just do the real computation.
    if key.param_env.reveal() == Reveal::All {
        let mut key = key;
        key.param_env = key.param_env.with_user_facing();
        match tcx.eval_to_allocation_raw(key) {
            // try again with reveal all as requested
            Err(ErrorHandled::TooGeneric) => {}
            // deduplicate calls
            other => return other,
        }
    }
    if cfg!(debug_assertions) {
        // Make sure we format the instance even if we do not print it.
        // This serves as a regression test against an ICE on printing.
        // The next two lines concatenated contain some discussion:
        // https://rust-lang.zulipchat.com/#narrow/stream/146212-t-compiler.2Fconst-eval/
        // subject/anon_const_instance_printing/near/135980032
        let instance = with_no_trimmed_paths!(key.value.instance.to_string());
        trace!("const eval: {:?} ({})", key, instance);
    }

    let cid = key.value;
    let def = cid.instance.def.def_id();
    let is_static = tcx.is_static(def);

    let mut ecx = InterpCx::new(
        tcx,
        tcx.def_span(def),
        key.param_env,
        // Statics (and promoteds inside statics) may access other statics, because unlike consts
        // they do not have to behave "as if" they were evaluated at runtime.
        CompileTimeInterpreter::new(
            CanAccessStatics::from(is_static),
            if tcx.sess.opts.unstable_opts.extra_const_ub_checks {
                CheckAlignment::Error
            } else {
                CheckAlignment::FutureIncompat
            },
        ),
    );

    let res = ecx.load_mir(cid.instance.def, cid.promoted);
    match res.and_then(|body| eval_body_using_ecx(&mut ecx, cid, &body)) {
        Err(error) => {
            let (error, backtrace) = error.into_parts();
            backtrace.print_backtrace();

            let (kind, instance) = if is_static {
                ("static", String::new())
            } else {
                // If the current item has generics, we'd like to enrich the message with the
                // instance and its args: to show the actual compile-time values, in addition to
                // the expression, leading to the const eval error.
                let instance = &key.value.instance;
                if !instance.args.is_empty() {
                    let instance = with_no_trimmed_paths!(instance.to_string());
                    ("const_with_path", instance)
                } else {
                    ("const", String::new())
                }
            };

            Err(super::report(
                *ecx.tcx,
                error,
                None,
                || super::get_span_and_frames(&ecx),
                |span, frames| ConstEvalError {
                    span,
                    error_kind: kind,
                    instance,
                    frame_notes: frames,
                },
            ))
        }
        Ok(mplace) => {
            // Since evaluation had no errors, validate the resulting constant.
            // This is a separate `try` block to provide more targeted error reporting.
            let validation: Result<_, InterpErrorInfo<'_>> = try {
                let mut ref_tracking = RefTracking::new(mplace);
                let mut inner = false;
                while let Some((mplace, path)) = ref_tracking.todo.pop() {
                    let mode = match tcx.static_mutability(cid.instance.def_id()) {
                        Some(_) if cid.promoted.is_some() => {
                            // Promoteds in statics are allowed to point to statics.
                            CtfeValidationMode::Const { inner, allow_static_ptrs: true }
                        }
                        Some(_) => CtfeValidationMode::Regular, // a `static`
                        None => CtfeValidationMode::Const { inner, allow_static_ptrs: false },
                    };
                    ecx.const_validate_operand(&mplace.into(), path, &mut ref_tracking, mode)?;
                    inner = true;
                }
            };
            let alloc_id = mplace.ptr.provenance.unwrap();

            // Validation failed, report an error. This is always a hard error.
            if let Err(error) = validation {
                let (error, backtrace) = error.into_parts();
                backtrace.print_backtrace();

                let ub_note = matches!(error, InterpError::UndefinedBehavior(_)).then(|| {});

                let alloc = ecx.tcx.global_alloc(alloc_id).unwrap_memory().inner();
                let mut bytes = String::new();
                if alloc.size() != abi::Size::ZERO {
                    bytes = "\n".into();
                    // FIXME(translation) there might be pieces that are translatable.
                    write_allocation_bytes(*ecx.tcx, alloc, &mut bytes, "    ").unwrap();
                }
                let raw_bytes = errors::RawBytesNote {
                    size: alloc.size().bytes(),
                    align: alloc.align.bytes(),
                    bytes,
                };

                Err(super::report(
                    *ecx.tcx,
                    error,
                    None,
                    || super::get_span_and_frames(&ecx),
                    move |span, frames| errors::UndefinedBehavior {
                        span,
                        ub_note,
                        frames,
                        raw_bytes,
                    },
                ))
            } else {
                // Convert to raw constant
                Ok(ConstAlloc { alloc_id, ty: mplace.layout.ty })
            }
        }
    }
}
