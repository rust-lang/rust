use super::{CompileTimeEvalContext, CompileTimeInterpreter, ConstEvalErr, MemoryExtra};
use crate::interpret::eval_nullary_intrinsic;
use crate::interpret::{
    intern_const_alloc_recursive, Allocation, ConstAlloc, ConstValue, CtfeValidationMode, GlobalId,
    Immediate, InternKind, InterpCx, InterpResult, MPlaceTy, MemoryKind, OpTy, RefTracking, Scalar,
    ScalarMaybeUninit, StackPopCleanup,
};

use rustc_hir::def::DefKind;
use rustc_middle::mir;
use rustc_middle::mir::interpret::ErrorHandled;
use rustc_middle::traits::Reveal;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{self, subst::Subst, TyCtxt};
use rustc_span::source_map::Span;
use rustc_target::abi::{Abi, LayoutOf};
use std::convert::{TryFrom, TryInto};

pub fn note_on_undefined_behavior_error() -> &'static str {
    "The rules on what exactly is undefined behavior aren't clear, \
     so this check might be overzealous. Please open an issue on the rustc \
     repository if you believe it should not be considered undefined behavior."
}

// Returns a pointer to where the result lives
fn eval_body_using_ecx<'mir, 'tcx>(
    ecx: &mut CompileTimeEvalContext<'mir, 'tcx>,
    cid: GlobalId<'tcx>,
    body: &'mir mir::Body<'tcx>,
) -> InterpResult<'tcx, MPlaceTy<'tcx>> {
    debug!("eval_body_using_ecx: {:?}, {:?}", cid, ecx.param_env);
    let tcx = *ecx.tcx;
    let layout = ecx.layout_of(body.return_ty().subst(tcx, cid.instance.substs))?;
    assert!(!layout.is_unsized());
    let ret = ecx.allocate(layout, MemoryKind::Stack);

    let name =
        with_no_trimmed_paths(|| ty::tls::with(|tcx| tcx.def_path_str(cid.instance.def_id())));
    let prom = cid.promoted.map_or(String::new(), |p| format!("::promoted[{:?}]", p));
    trace!("eval_body_using_ecx: pushing stack frame for global: {}{}", name, prom);

    // Assert all args (if any) are zero-sized types; `eval_body_using_ecx` doesn't
    // make sense if the body is expecting nontrivial arguments.
    // (The alternative would be to use `eval_fn_call` with an args slice.)
    for arg in body.args_iter() {
        let decl = body.local_decls.get(arg).expect("arg missing from local_decls");
        let layout = ecx.layout_of(decl.ty.subst(tcx, cid.instance.substs))?;
        assert!(layout.is_zst())
    }

    ecx.push_stack_frame(
        cid.instance,
        body,
        Some(ret.into()),
        StackPopCleanup::None { cleanup: false },
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
    intern_const_alloc_recursive(ecx, intern_kind, ret);

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
        CompileTimeInterpreter::new(tcx.sess.const_eval_limit()),
        MemoryExtra { can_access_statics },
    )
}

/// This function converts an interpreter value into a constant that is meant for use in the
/// type system.
pub(super) fn op_to_const<'tcx>(
    ecx: &CompileTimeEvalContext<'_, 'tcx>,
    op: OpTy<'tcx>,
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
        op.try_as_mplace(ecx)
    };

    let to_const_value = |mplace: MPlaceTy<'_>| match mplace.ptr {
        Scalar::Ptr(ptr) => {
            let alloc = ecx.tcx.global_alloc(ptr.alloc_id).unwrap_memory();
            ConstValue::ByRef { alloc, offset: ptr.offset }
        }
        Scalar::Raw { data, .. } => {
            assert!(mplace.layout.is_zst());
            assert_eq!(
                u64::try_from(data).unwrap() % mplace.layout.align.abi.bytes(),
                0,
                "this MPlaceTy must come from a validated constant, thus we can assume the \
                alignment is correct",
            );
            ConstValue::Scalar(Scalar::zst())
        }
    };
    match immediate {
        Ok(mplace) => to_const_value(mplace),
        // see comment on `let try_as_immediate` above
        Err(imm) => match *imm {
            Immediate::Scalar(x) => match x {
                ScalarMaybeUninit::Scalar(s) => ConstValue::Scalar(s),
                ScalarMaybeUninit::Uninit => to_const_value(op.assert_mem_place(ecx)),
            },
            Immediate::ScalarPair(a, b) => {
                let (data, start) = match a.check_init().unwrap() {
                    Scalar::Ptr(ptr) => {
                        (ecx.tcx.global_alloc(ptr.alloc_id).unwrap_memory(), ptr.offset.bytes())
                    }
                    Scalar::Raw { .. } => (
                        ecx.tcx
                            .intern_const_alloc(Allocation::from_byte_aligned_bytes(b"" as &[u8])),
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
    op_to_const(&ecx, mplace.into())
}

pub fn eval_to_const_value_raw_provider<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: ty::ParamEnvAnd<'tcx, GlobalId<'tcx>>,
) -> ::rustc_middle::mir::interpret::EvalToConstValueResult<'tcx> {
    // see comment in const_eval_raw_provider for what we're doing here
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
        let substs = match ty.kind() {
            ty::FnDef(_, substs) => substs,
            _ => bug!("intrinsic with type {:?}", ty),
        };
        return eval_nullary_intrinsic(tcx, key.param_env, def_id, substs).map_err(|error| {
            let span = tcx.def_span(def_id);
            let error = ConstEvalErr { error: error.kind, stacktrace: vec![], span };
            error.report_as_error(tcx.at(span), "could not evaluate nullary intrinsic")
        });
    }

    tcx.eval_to_allocation_raw(key).map(|val| turn_into_const_value(tcx, val, key))
}

pub fn eval_to_allocation_raw_provider<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: ty::ParamEnvAnd<'tcx, GlobalId<'tcx>>,
) -> ::rustc_middle::mir::interpret::EvalToAllocationRawResult<'tcx> {
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
        let instance = with_no_trimmed_paths(|| key.value.instance.to_string());
        trace!("const eval: {:?} ({})", key, instance);
    }

    let cid = key.value;
    let def = cid.instance.def.with_opt_param();

    if let Some(def) = def.as_local() {
        if tcx.has_typeck_results(def.did) {
            if let Some(error_reported) = tcx.typeck_opt_const_arg(def).tainted_by_errors {
                return Err(ErrorHandled::Reported(error_reported));
            }
        }
    }

    let is_static = tcx.is_static(def.did);

    let mut ecx = InterpCx::new(
        tcx,
        tcx.def_span(def.did),
        key.param_env,
        CompileTimeInterpreter::new(tcx.sess.const_eval_limit()),
        MemoryExtra { can_access_statics: is_static },
    );

    let res = ecx.load_mir(cid.instance.def, cid.promoted);
    match res.and_then(|body| eval_body_using_ecx(&mut ecx, cid, &body)) {
        Err(error) => {
            let err = ConstEvalErr::new(&ecx, error, None);
            // errors in statics are always emitted as fatal errors
            if is_static {
                // Ensure that if the above error was either `TooGeneric` or `Reported`
                // an error must be reported.
                let v = err.report_as_error(
                    ecx.tcx.at(ecx.cur_span()),
                    "could not evaluate static initializer",
                );

                // If this is `Reveal:All`, then we need to make sure an error is reported but if
                // this is `Reveal::UserFacing`, then it's expected that we could get a
                // `TooGeneric` error. When we fall back to `Reveal::All`, then it will either
                // succeed or we'll report this error then.
                if key.param_env.reveal() == Reveal::All {
                    tcx.sess.delay_span_bug(
                        err.span,
                        &format!("static eval failure did not emit an error: {:#?}", v),
                    );
                }

                Err(v)
            } else if let Some(def) = def.as_local() {
                // constant defined in this crate, we can figure out a lint level!
                match tcx.def_kind(def.did.to_def_id()) {
                    // constants never produce a hard error at the definition site. Anything else is
                    // a backwards compatibility hazard (and will break old versions of winapi for
                    // sure)
                    //
                    // note that validation may still cause a hard error on this very same constant,
                    // because any code that existed before validation could not have failed
                    // validation thus preventing such a hard error from being a backwards
                    // compatibility hazard
                    DefKind::Const | DefKind::AssocConst => {
                        let hir_id = tcx.hir().local_def_id_to_hir_id(def.did);
                        Err(err.report_as_lint(
                            tcx.at(tcx.def_span(def.did)),
                            "any use of this value will cause an error",
                            hir_id,
                            Some(err.span),
                        ))
                    }
                    // promoting runtime code is only allowed to error if it references broken
                    // constants any other kind of error will be reported to the user as a
                    // deny-by-default lint
                    _ => {
                        if let Some(p) = cid.promoted {
                            let span = tcx.promoted_mir_opt_const_arg(def.to_global())[p].span;
                            if let err_inval!(ReferencedConstant) = err.error {
                                Err(err.report_as_error(
                                    tcx.at(span),
                                    "evaluation of constant expression failed",
                                ))
                            } else {
                                Err(err.report_as_lint(
                                    tcx.at(span),
                                    "reaching this expression at runtime will panic or abort",
                                    tcx.hir().local_def_id_to_hir_id(def.did),
                                    Some(err.span),
                                ))
                            }
                        // anything else (array lengths, enum initializers, constant patterns) are
                        // reported as hard errors
                        } else {
                            Err(err.report_as_error(
                                ecx.tcx.at(ecx.cur_span()),
                                "evaluation of constant value failed",
                            ))
                        }
                    }
                }
            } else {
                // use of broken constant from other crate
                Err(err.report_as_error(ecx.tcx.at(ecx.cur_span()), "could not evaluate constant"))
            }
        }
        Ok(mplace) => {
            // Since evaluation had no errors, valiate the resulting constant:
            let validation = try {
                // FIXME do not validate promoteds until a decision on
                // https://github.com/rust-lang/rust/issues/67465 and
                // https://github.com/rust-lang/rust/issues/67534 is made.
                // Promoteds can contain unexpected `UnsafeCell` and reference `static`s, but their
                // otherwise restricted form ensures that this is still sound. We just lose the
                // extra safety net of some of the dynamic checks. They can also contain invalid
                // values, but since we do not usually check intermediate results of a computation
                // for validity, it might be surprising to do that here.
                if cid.promoted.is_none() {
                    let mut ref_tracking = RefTracking::new(mplace);
                    let mut inner = false;
                    while let Some((mplace, path)) = ref_tracking.todo.pop() {
                        let mode = match tcx.static_mutability(cid.instance.def_id()) {
                            Some(_) => CtfeValidationMode::Regular, // a `static`
                            None => CtfeValidationMode::Const { inner },
                        };
                        ecx.const_validate_operand(mplace.into(), path, &mut ref_tracking, mode)?;
                        inner = true;
                    }
                }
            };
            if let Err(error) = validation {
                // Validation failed, report an error
                let err = ConstEvalErr::new(&ecx, error, None);
                Err(err.struct_error(
                    ecx.tcx,
                    "it is undefined behavior to use this value",
                    |mut diag| {
                        diag.note(note_on_undefined_behavior_error());
                        diag.emit();
                    },
                ))
            } else {
                // Convert to raw constant
                Ok(ConstAlloc { alloc_id: mplace.ptr.assert_ptr().alloc_id, ty: mplace.layout.ty })
            }
        }
    }
}
