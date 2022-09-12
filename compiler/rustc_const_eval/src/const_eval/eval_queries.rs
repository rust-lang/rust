use super::{CompileTimeEvalContext, CompileTimeInterpreter, ConstEvalErr};
use crate::interpret::eval_nullary_intrinsic;
use crate::interpret::{
    intern_const_alloc_recursive, Allocation, ConstAlloc, ConstValue, CtfeValidationMode, GlobalId,
    Immediate, InternKind, InterpCx, InterpError, InterpResult, MPlaceTy, MemoryKind, OpTy,
    RefTracking, StackPopCleanup,
};

use rustc_hir::def::DefKind;
use rustc_middle::mir;
use rustc_middle::mir::interpret::ErrorHandled;
use rustc_middle::mir::pretty::display_allocation;
use rustc_middle::traits::Reveal;
use rustc_middle::ty::layout::LayoutOf;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{self, subst::Subst, TyCtxt};
use rustc_span::source_map::Span;
use rustc_target::abi::{self, Abi};
use std::borrow::Cow;
use std::convert::TryInto;

const NOTE_ON_UNDEFINED_BEHAVIOR_ERROR: &str = "The rules on what exactly is undefined behavior aren't clear, \
     so this check might be overzealous. Please open an issue on the rustc \
     repository if you believe it should not be considered undefined behavior.";

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
    let layout = ecx.layout_of(body.bound_return_ty().subst(tcx, cid.instance.substs))?;
    assert!(!layout.is_unsized());
    let ret = ecx.allocate(layout, MemoryKind::Stack)?;

    trace!(
        "eval_body_using_ecx: pushing stack frame for global: {}{}",
        with_no_trimmed_paths!(ty::tls::with(|tcx| tcx.def_path_str(cid.instance.def_id()))),
        cid.promoted.map_or_else(String::new, |p| format!("::promoted[{:?}]", p))
    );

    ecx.push_stack_frame(
        cid.instance,
        body,
        &ret.into(),
        StackPopCleanup::Root { cleanup: false },
    )?;

    // The main interpreter loop.
    ecx.run()?;

    // Intern the result
    let intern_kind = if cid.promoted.is_some() {
        InternKind::Promoted
    } else {
        match tcx.static_mutability(cid.instance.def_id()) {
            Some(m) => InternKind::Static(m),
            None => InternKind::Constant,
        }
    };
    ecx.machine.check_alignment = false; // interning doesn't need to respect alignment
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
    can_access_statics: bool,
) -> CompileTimeEvalContext<'mir, 'tcx> {
    debug!("mk_eval_cx: {:?}", param_env);
    InterpCx::new(
        tcx,
        root_span,
        param_env,
        CompileTimeInterpreter::new(
            tcx.const_eval_limit(),
            can_access_statics,
            /*check_alignment:*/ false,
        ),
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
        Err(ecx.read_immediate(op).expect("normalization works on validated constants"))
    } else {
        // It is guaranteed that any non-slice scalar pair is actually ByRef here.
        // When we come back from raw const eval, we are always by-ref. The only way our op here is
        // by-val is if we are in destructure_mir_constant, i.e., if this is (a field of) something that we
        // "tried to make immediate" before. We wouldn't do that for non-slice scalar pairs or
        // structs containing such.
        op.try_as_mplace()
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
        Ok(ref mplace) => to_const_value(mplace),
        // see comment on `let try_as_immediate` above
        Err(imm) => match *imm {
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
                        ecx.tcx.intern_const_alloc(Allocation::from_bytes_byte_aligned_immutable(
                            b"" as &[u8],
                        )),
                        0,
                    ),
                };
                let len = b.to_machine_usize(ecx).unwrap();
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
    // This is just accessing an already computed constant, so no need to check alginment here.
    let ecx = mk_eval_cx(
        tcx,
        tcx.def_span(key.value.instance.def_id()),
        key.param_env,
        /*can_access_statics:*/ is_static,
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
        let ty::FnDef(_, substs) = ty.kind() else {
            bug!("intrinsic with type {:?}", ty);
        };
        return eval_nullary_intrinsic(tcx, key.param_env, def_id, substs).map_err(|error| {
            let span = tcx.def_span(def_id);
            let error = ConstEvalErr { error: error.into_kind(), stacktrace: vec![], span };
            error.report_as_error(tcx.at(span), "could not evaluate nullary intrinsic")
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
    let def = cid.instance.def.with_opt_param();
    let is_static = tcx.is_static(def.did);

    let mut ecx = InterpCx::new(
        tcx,
        tcx.def_span(def.did),
        key.param_env,
        // Statics (and promoteds inside statics) may access other statics, because unlike consts
        // they do not have to behave "as if" they were evaluated at runtime.
        CompileTimeInterpreter::new(
            tcx.const_eval_limit(),
            /*can_access_statics:*/ is_static,
            /*check_alignment:*/ tcx.sess.opts.unstable_opts.extra_const_ub_checks,
        ),
    );

    let res = ecx.load_mir(cid.instance.def, cid.promoted);
    match res.and_then(|body| eval_body_using_ecx(&mut ecx, cid, &body)) {
        Err(error) => {
            let err = ConstEvalErr::new(&ecx, error, None);
            // Some CTFE errors raise just a lint, not a hard error; see
            // <https://github.com/rust-lang/rust/issues/71800>.
            let is_hard_err = if let Some(def) = def.as_local() {
                // (Associated) consts only emit a lint, since they might be unused.
                !matches!(tcx.def_kind(def.did.to_def_id()), DefKind::Const | DefKind::AssocConst)
                    // check if the inner InterpError is hard
                    || err.error.is_hard_err()
            } else {
                // use of broken constant from other crate: always an error
                true
            };

            if is_hard_err {
                let msg = if is_static {
                    Cow::from("could not evaluate static initializer")
                } else {
                    // If the current item has generics, we'd like to enrich the message with the
                    // instance and its substs: to show the actual compile-time values, in addition to
                    // the expression, leading to the const eval error.
                    let instance = &key.value.instance;
                    if !instance.substs.is_empty() {
                        let instance = with_no_trimmed_paths!(instance.to_string());
                        let msg = format!("evaluation of `{}` failed", instance);
                        Cow::from(msg)
                    } else {
                        Cow::from("evaluation of constant value failed")
                    }
                };

                Err(err.report_as_error(ecx.tcx.at(err.span), &msg))
            } else {
                let hir_id = tcx.hir().local_def_id_to_hir_id(def.as_local().unwrap().did);
                Err(err.report_as_lint(
                    tcx.at(tcx.def_span(def.did)),
                    "any use of this value will cause an error",
                    hir_id,
                    Some(err.span),
                ))
            }
        }
        Ok(mplace) => {
            // Since evaluation had no errors, validate the resulting constant.
            // This is a separate `try` block to provide more targeted error reporting.
            let validation = try {
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
            if let Err(error) = validation {
                // Validation failed, report an error. This is always a hard error.
                let err = ConstEvalErr::new(&ecx, error, None);
                Err(err.struct_error(
                    ecx.tcx,
                    "it is undefined behavior to use this value",
                    |diag| {
                        if matches!(err.error, InterpError::UndefinedBehavior(_)) {
                            diag.note(NOTE_ON_UNDEFINED_BEHAVIOR_ERROR);
                        }
                        diag.note(&format!(
                            "the raw bytes of the constant ({}",
                            display_allocation(
                                *ecx.tcx,
                                ecx.tcx.global_alloc(alloc_id).unwrap_memory().inner()
                            )
                        ));
                    },
                ))
            } else {
                // Convert to raw constant
                Ok(ConstAlloc { alloc_id, ty: mplace.layout.ty })
            }
        }
    }
}
