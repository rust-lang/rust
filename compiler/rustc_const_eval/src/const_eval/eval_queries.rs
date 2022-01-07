use super::{CompileTimeEvalContext, CompileTimeInterpreter, ConstEvalError, MemoryExtra};
use crate::interpret::eval_nullary_intrinsic;
use crate::interpret::{
    intern_const_alloc_recursive, Allocation, ConstAlloc, ConstValue, CtfeValidationMode, GlobalId,
    Immediate, InternKind, InterpCx, InterpResult, MPlaceTy, MemoryKind, OpTy, RefTracking, Scalar,
    ScalarMaybeUninit, StackPopCleanup,
};

use rustc_errors::ErrorReported;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_middle::mir;
use rustc_middle::mir::interpret::{
    ConstDedupError, ConstDedupResult, ConstErrorEmitted, ConstEvalErr, ConstOrigin, ErrorHandled,
};
use rustc_middle::mir::pretty::display_allocation;
use rustc_middle::traits::Reveal;
use rustc_middle::ty::layout::{LayoutError, LayoutOf};
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{self, subst::Subst, TyCtxt};
use rustc_span::source_map::Span;
use rustc_target::abi::Abi;
use std::convert::TryInto;

pub fn note_on_undefined_behavior_error() -> &'static str {
    "The rules on what exactly is undefined behavior aren't clear, \
     so this check might be overzealous. Please open an issue on the rustc \
     repository if you believe it should not be considered undefined behavior."
}

// Returns a pointer to where the result lives
#[instrument(skip(ecx, body), level = "debug")]
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
                    | DefKind::Static
                    | DefKind::ConstParam
                    | DefKind::AnonConst
                    | DefKind::InlineConst
                    | DefKind::AssocConst
            ),
        "Unexpected DefKind: {:?}",
        ecx.tcx.def_kind(cid.instance.def_id())
    );
    let layout = ecx.layout_of(body.return_ty().subst(tcx, cid.instance.substs))?;
    assert!(!layout.is_unsized());
    let ret = ecx.allocate(layout, MemoryKind::Stack)?;

    trace!(
        "eval_body_using_ecx: pushing stack frame for global: {}{}",
        with_no_trimmed_paths(|| ty::tls::with(|tcx| tcx.def_path_str(cid.instance.def_id()))),
        cid.promoted.map_or_else(String::new, |p| format!("::promoted[{:?}]", p))
    );

    ecx.push_stack_frame(
        cid.instance,
        body,
        Some(&ret.into()),
        StackPopCleanup::None { cleanup: false },
    )?;

    debug!("returned from push_stack_frame");

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
    intern_const_alloc_recursive(ecx, intern_kind, &ret)?;

    debug!("eval_body_using_ecx done: {:?}", *ret);
    Ok(ret)
}

/// The `InterpCx` is only meant to be used to do field and index projections into constants for
/// `simd_shuffle` and const patterns in match arms.
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
        CompileTimeInterpreter::new(tcx.const_eval_limit()),
        MemoryExtra { can_access_statics },
    )
}

/// This function converts an interpreter value into a constant that is meant for use in the
/// type system.
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
        Abi::Scalar(..) => true,
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
        // by-val is if we are in destructure_const, i.e., if this is (a field of) something that we
        // "tried to make immediate" before. We wouldn't do that for non-slice scalar pairs or
        // structs containing such.
        op.try_as_mplace()
    };

    // We know `offset` is relative to the allocation, so we can use `into_parts`.
    let to_const_value = |mplace: &MPlaceTy<'_>| match mplace.ptr.into_parts() {
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
            ConstValue::Scalar(Scalar::ZST)
        }
    };

    match immediate {
        Ok(ref mplace) => to_const_value(mplace),
        // see comment on `let try_as_immediate` above
        Err(imm) => match *imm {
            Immediate::Scalar(x) => match x {
                ScalarMaybeUninit::Scalar(s) => ConstValue::Scalar(s),
                ScalarMaybeUninit::Uninit => to_const_value(&op.assert_mem_place()),
            },
            Immediate::ScalarPair(a, b) => {
                // We know `offset` is relative to the allocation, so we can use `into_parts`.
                let (data, start) = match ecx.scalar_to_ptr(a.check_init().unwrap()).into_parts() {
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
        },
    }
}

fn turn_into_const_value<'tcx>(
    tcx: TyCtxt<'tcx>,
    constant: ConstAlloc<'tcx>,
    key: ty::ParamEnvAnd<'tcx, GlobalId<'tcx>>,
) -> ConstValue<'tcx> {
    let cid = key.value;
    let def_id = cid.instance.def.def_id();
    let is_static = tcx.is_static(def_id);
    let ecx = mk_eval_cx(tcx, tcx.def_span(key.value.instance.def_id()), key.param_env, is_static);

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
    assert!(key.param_env.constness() == hir::Constness::Const);
    let (param_env, id) = key.into_parts();
    let reveal = param_env.reveal();

    // We call `const_eval` for zero arg intrinsics, too, in order to cache their value.
    // Catch such calls and evaluate them instead of trying to load a constant's MIR.
    if let ty::InstanceDef::Intrinsic(def_id) = key.value.instance.def {
        let ty = key.value.instance.ty(tcx, param_env);
        let substs = match ty.kind() {
            ty::FnDef(_, substs) => substs,
            _ => bug!("intrinsic with type {:?}", ty),
        };
        match eval_nullary_intrinsic(tcx, param_env, def_id, substs) {
            Ok(val) => {
                // store result for deduplication
                let res = ConstDedupResult::new(reveal, Ok(val), None);
                tcx.save_const_value_for_dedup(id, res);

                return Ok(val);
            }
            Err(e) => {
                let span = tcx.def_span(def_id);
                let error = ConstEvalErr { error: e.into_kind(), stacktrace: vec![], span };

                let error_handled = tcx.handle_err_for_dedup(
                    id,
                    ConstOrigin::ConstValue,
                    error,
                    reveal,
                    span,
                    |e| {
                        tcx.report_and_add_error(
                            id,
                            e,
                            span,
                            "could not evaluate nullary intrinsic",
                        )
                    },
                );

                return Err(error_handled);
            }
        }
    }

    let result =
        tcx.dedup_eval_alloc_raw(key, None).map(|val| turn_into_const_value(tcx, val, key));

    // store result for deduplication
    let val = ConstDedupResult::new(
        reveal,
        result.map_err(|e| ConstDedupError::new_handled(e, reveal)),
        None,
    );
    tcx.save_const_value_for_dedup(id, val);

    result
}

#[instrument(skip(tcx), level = "debug")]
pub fn eval_to_allocation_raw_provider<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: ty::ParamEnvAnd<'tcx, GlobalId<'tcx>>,
) -> ::rustc_middle::mir::interpret::EvalToAllocationRawResult<'tcx> {
    assert!(key.param_env.constness() == hir::Constness::Const);
    let (param_env, cid) = key.into_parts();
    let reveal = param_env.reveal();

    if cfg!(debug_assertions) {
        // Make sure we format the instance even if we do not print it.
        // This serves as a regression test against an ICE on printing.
        // The next two lines concatenated contain some discussion:
        // https://rust-lang.zulipchat.com/#narrow/stream/146212-t-compiler.2Fconst-eval/
        // subject/anon_const_instance_printing/near/135980032
        let instance = with_no_trimmed_paths(|| cid.instance.to_string());
        trace!("const eval: {:?} ({})", key, instance);
    }

    let def = cid.instance.def.with_opt_param();

    if let Some(def) = def.as_local() {
        if tcx.has_typeck_results(def.did) {
            if let Some(error_reported) = tcx.typeck_opt_const_arg(def).tainted_by_errors {
                let err = ConstErrorEmitted::Emitted(ErrorHandled::Reported(error_reported));
                let err_handled =
                    tcx.handle_reported_error_for_dedup(cid, ConstOrigin::Alloc, err, reveal);

                return Err(err_handled);
            }
        }
        if !tcx.is_mir_available(def.did) {
            tcx.sess.delay_span_bug(
                tcx.def_span(def.did),
                &format!("no MIR body is available for {:?}", def.did),
            );

            let err = ConstErrorEmitted::Emitted(ErrorHandled::Reported(ErrorReported {}));
            let err_handled =
                tcx.handle_reported_error_for_dedup(cid, ConstOrigin::Alloc, err, reveal);

            return Err(err_handled);
        }
        if let Some(error_reported) = tcx.mir_const_qualif_opt_const_arg(def).error_occured {
            let err = ConstErrorEmitted::Emitted(ErrorHandled::Reported(error_reported));
            let err_handled =
                tcx.handle_reported_error_for_dedup(cid, ConstOrigin::Alloc, err, reveal);

            return Err(err_handled);
        }
    }

    let is_static = tcx.is_static(def.did);

    let mut ecx = InterpCx::new(
        tcx,
        tcx.def_span(def.did),
        key.param_env,
        CompileTimeInterpreter::new(tcx.const_eval_limit()),
        // Statics (and promoteds inside statics) may access other statics, because unlike consts
        // they do not have to behave "as if" they were evaluated at runtime.
        MemoryExtra { can_access_statics: is_static },
    );

    let res = ecx.load_mir(cid.instance.def, cid.promoted);
    match res.and_then(|body| eval_body_using_ecx(&mut ecx, cid, &body)) {
        Err(error) => {
            debug!("error from eval_body_using_ecx: {:?}", error);
            if reveal == Reveal::Selection {
                match error.kind() {
                    err_inval!(Layout(LayoutError::Unknown(_)))
                    | err_inval!(TooGeneric)
                    | err_inval!(AlreadyReported(_)) => {
                        // We do want to report these errors
                    }
                    _ => {
                        let err = ConstEvalError::new(&ecx, error, None);
                        let error_handled = tcx.handle_err_for_dedup(
                            cid,
                            ConstOrigin::Alloc,
                            err.into_inner(),
                            reveal,
                            ecx.cur_span(),
                            |_e| ErrorHandled::Silent(cid),
                        );

                        return Err(error_handled);
                    }
                }
            }

            let err = ConstEvalError::new(&ecx, error, None);
            let err_handled = tcx.handle_err_for_dedup(
                cid,
                ConstOrigin::Alloc,
                err.into_inner(),
                reveal,
                ecx.cur_span(),
                |e| tcx.report_alloc_error(cid, param_env, e, is_static, def, ecx.cur_span()),
            );

            Err(err_handled)
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
                let err = ConstEvalError::new(&ecx, error, None).into_inner();

                // FIXME: Do we also want to keep these silent with Reveal::Selection?
                let error_handled = tcx.handle_err_for_dedup(
                    cid,
                    ConstOrigin::Alloc,
                    err,
                    reveal,
                    ecx.cur_span(),
                    |e| {
                        e.struct_error(
                            ecx.tcx,
                            "it is undefined behavior to use this value",
                            |mut diag| {
                                diag.note(note_on_undefined_behavior_error());
                                diag.note(&format!(
                                    "the raw bytes of the constant ({}",
                                    display_allocation(
                                        *ecx.tcx,
                                        ecx.tcx.global_alloc(alloc_id).unwrap_memory()
                                    )
                                ));
                                diag.emit();
                            },
                        )
                        .get_error()
                    },
                );

                Err(error_handled)
            } else {
                // Convert to raw constant
                let const_alloc = ConstAlloc { alloc_id, ty: mplace.layout.ty };
                let result = Ok(const_alloc);
                let val = ConstDedupResult::new(reveal, result, None);

                // store result in order to deduplicate later
                tcx.save_alloc_for_dedup(cid, val);

                Ok(const_alloc)
            }
        }
    }
}
