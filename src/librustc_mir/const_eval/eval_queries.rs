use super::{error_to_const_error, CompileTimeEvalContext, CompileTimeInterpreter, MemoryExtra};
use crate::interpret::eval_nullary_intrinsic;
use crate::interpret::{
    intern_const_alloc_recursive, Allocation, ConstValue, GlobalId, ImmTy, Immediate, InternKind,
    InterpCx, InterpResult, MPlaceTy, MemoryKind, OpTy, RawConst, RefTracking, Scalar,
    ScalarMaybeUndef, StackPopCleanup,
};
use rustc::mir;
use rustc::mir::interpret::{ConstEvalErr, ErrorHandled};
use rustc::traits::Reveal;
use rustc::ty::{self, layout, layout::LayoutOf, subst::Subst, TyCtxt};
use rustc_hir::def::DefKind;
use rustc_span::source_map::Span;
use std::convert::TryInto;

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
    let tcx = ecx.tcx.tcx;
    let layout = ecx.layout_of(body.return_ty().subst(tcx, cid.instance.substs))?;
    assert!(!layout.is_unsized());
    let ret = ecx.allocate(layout, MemoryKind::Stack);

    let name = ty::tls::with(|tcx| tcx.def_path_str(cid.instance.def_id()));
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
        body.span,
        body,
        Some(ret.into()),
        StackPopCleanup::None { cleanup: false },
    )?;

    // The main interpreter loop.
    ecx.run()?;

    // Intern the result
    let intern_kind = match tcx.static_mutability(cid.instance.def_id()) {
        Some(m) => InternKind::Static(m),
        None if cid.promoted.is_some() => InternKind::Promoted,
        _ => InternKind::Constant,
    };
    intern_const_alloc_recursive(
        ecx,
        intern_kind,
        ret,
        body.ignore_interior_mut_in_const_validation,
    )?;

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
    span: Span,
    param_env: ty::ParamEnv<'tcx>,
    can_access_statics: bool,
) -> CompileTimeEvalContext<'mir, 'tcx> {
    debug!("mk_eval_cx: {:?}", param_env);
    InterpCx::new(
        tcx.at(span),
        param_env,
        CompileTimeInterpreter::new(),
        MemoryExtra { can_access_statics },
    )
}

pub(super) fn op_to_const<'tcx>(
    ecx: &CompileTimeEvalContext<'_, 'tcx>,
    op: OpTy<'tcx>,
) -> &'tcx ty::Const<'tcx> {
    // We do not have value optimizations for everything.
    // Only scalars and slices, since they are very common.
    // Note that further down we turn scalars of undefined bits back to `ByRef`. These can result
    // from scalar unions that are initialized with one of their zero sized variants. We could
    // instead allow `ConstValue::Scalar` to store `ScalarMaybeUndef`, but that would affect all
    // the usual cases of extracting e.g. a `usize`, without there being a real use case for the
    // `Undef` situation.
    let try_as_immediate = match op.layout.abi {
        layout::Abi::Scalar(..) => true,
        layout::Abi::ScalarPair(..) => match op.layout.ty.kind {
            ty::Ref(_, inner, _) => match inner.kind {
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
        // by-val is if we are in const_field, i.e., if this is (a field of) something that we
        // "tried to make immediate" before. We wouldn't do that for non-slice scalar pairs or
        // structs containing such.
        op.try_as_mplace(ecx)
    };

    let to_const_value = |mplace: MPlaceTy<'_>| match mplace.ptr {
        Scalar::Ptr(ptr) => {
            let alloc = ecx.tcx.alloc_map.lock().unwrap_memory(ptr.alloc_id);
            ConstValue::ByRef { alloc, offset: ptr.offset }
        }
        Scalar::Raw { data, .. } => {
            assert!(mplace.layout.is_zst());
            assert_eq!(
                data,
                mplace.layout.align.abi.bytes().into(),
                "this MPlaceTy must come from `try_as_mplace` being used on a zst, so we know what
                 value this integer address must have",
            );
            ConstValue::Scalar(Scalar::zst())
        }
    };
    let val = match immediate {
        Ok(mplace) => to_const_value(mplace),
        // see comment on `let try_as_immediate` above
        Err(ImmTy { imm: Immediate::Scalar(x), .. }) => match x {
            ScalarMaybeUndef::Scalar(s) => ConstValue::Scalar(s),
            ScalarMaybeUndef::Undef => to_const_value(op.assert_mem_place(ecx)),
        },
        Err(ImmTy { imm: Immediate::ScalarPair(a, b), .. }) => {
            let (data, start) = match a.not_undef().unwrap() {
                Scalar::Ptr(ptr) => {
                    (ecx.tcx.alloc_map.lock().unwrap_memory(ptr.alloc_id), ptr.offset.bytes())
                }
                Scalar::Raw { .. } => (
                    ecx.tcx.intern_const_alloc(Allocation::from_byte_aligned_bytes(b"" as &[u8])),
                    0,
                ),
            };
            let len = b.to_machine_usize(&ecx.tcx.tcx).unwrap();
            let start = start.try_into().unwrap();
            let len: usize = len.try_into().unwrap();
            ConstValue::Slice { data, start, end: start + len }
        }
    };
    ecx.tcx.mk_const(ty::Const { val: ty::ConstKind::Value(val), ty: op.layout.ty })
}

fn validate_and_turn_into_const<'tcx>(
    tcx: TyCtxt<'tcx>,
    constant: RawConst<'tcx>,
    key: ty::ParamEnvAnd<'tcx, GlobalId<'tcx>>,
) -> ::rustc::mir::interpret::ConstEvalResult<'tcx> {
    let cid = key.value;
    let def_id = cid.instance.def.def_id();
    let is_static = tcx.is_static(def_id);
    let ecx = mk_eval_cx(tcx, tcx.def_span(key.value.instance.def_id()), key.param_env, is_static);
    let val = (|| {
        let mplace = ecx.raw_const_to_mplace(constant)?;

        // FIXME do not validate promoteds until a decision on
        // https://github.com/rust-lang/rust/issues/67465 is made
        if cid.promoted.is_none() {
            let mut ref_tracking = RefTracking::new(mplace);
            while let Some((mplace, path)) = ref_tracking.todo.pop() {
                ecx.validate_operand(mplace.into(), path, Some(&mut ref_tracking))?;
            }
        }
        // Now that we validated, turn this into a proper constant.
        // Statics/promoteds are always `ByRef`, for the rest `op_to_const` decides
        // whether they become immediates.
        if is_static || cid.promoted.is_some() {
            let ptr = mplace.ptr.assert_ptr();
            Ok(tcx.mk_const(ty::Const {
                val: ty::ConstKind::Value(ConstValue::ByRef {
                    alloc: ecx.tcx.alloc_map.lock().unwrap_memory(ptr.alloc_id),
                    offset: ptr.offset,
                }),
                ty: mplace.layout.ty,
            }))
        } else {
            Ok(op_to_const(&ecx, mplace.into()))
        }
    })();

    val.map_err(|error| {
        let err = error_to_const_error(&ecx, error);
        match err.struct_error(ecx.tcx, "it is undefined behavior to use this value", |mut diag| {
            diag.note(note_on_undefined_behavior_error());
            diag.emit();
        }) {
            Ok(_) => ErrorHandled::Reported,
            Err(err) => err,
        }
    })
}

pub fn const_eval_validated_provider<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: ty::ParamEnvAnd<'tcx, GlobalId<'tcx>>,
) -> ::rustc::mir::interpret::ConstEvalResult<'tcx> {
    // see comment in const_eval_raw_provider for what we're doing here
    if key.param_env.reveal == Reveal::All {
        let mut key = key;
        key.param_env.reveal = Reveal::UserFacing;
        match tcx.const_eval_validated(key) {
            // try again with reveal all as requested
            Err(ErrorHandled::TooGeneric) => {}
            // dedupliate calls
            other => return other,
        }
    }

    // We call `const_eval` for zero arg intrinsics, too, in order to cache their value.
    // Catch such calls and evaluate them instead of trying to load a constant's MIR.
    if let ty::InstanceDef::Intrinsic(def_id) = key.value.instance.def {
        let ty = key.value.instance.ty_env(tcx, key.param_env);
        let substs = match ty.kind {
            ty::FnDef(_, substs) => substs,
            _ => bug!("intrinsic with type {:?}", ty),
        };
        return eval_nullary_intrinsic(tcx, key.param_env, def_id, substs).map_err(|error| {
            let span = tcx.def_span(def_id);
            let error = ConstEvalErr { error: error.kind, stacktrace: vec![], span };
            error.report_as_error(tcx.at(span), "could not evaluate nullary intrinsic")
        });
    }

    tcx.const_eval_raw(key).and_then(|val| validate_and_turn_into_const(tcx, val, key))
}

pub fn const_eval_raw_provider<'tcx>(
    tcx: TyCtxt<'tcx>,
    key: ty::ParamEnvAnd<'tcx, GlobalId<'tcx>>,
) -> ::rustc::mir::interpret::ConstEvalRawResult<'tcx> {
    // Because the constant is computed twice (once per value of `Reveal`), we are at risk of
    // reporting the same error twice here. To resolve this, we check whether we can evaluate the
    // constant in the more restrictive `Reveal::UserFacing`, which most likely already was
    // computed. For a large percentage of constants that will already have succeeded. Only
    // associated constants of generic functions will fail due to not enough monomorphization
    // information being available.

    // In case we fail in the `UserFacing` variant, we just do the real computation.
    if key.param_env.reveal == Reveal::All {
        let mut key = key;
        key.param_env.reveal = Reveal::UserFacing;
        match tcx.const_eval_raw(key) {
            // try again with reveal all as requested
            Err(ErrorHandled::TooGeneric) => {}
            // dedupliate calls
            other => return other,
        }
    }
    if cfg!(debug_assertions) {
        // Make sure we format the instance even if we do not print it.
        // This serves as a regression test against an ICE on printing.
        // The next two lines concatenated contain some discussion:
        // https://rust-lang.zulipchat.com/#narrow/stream/146212-t-compiler.2Fconst-eval/
        // subject/anon_const_instance_printing/near/135980032
        let instance = key.value.instance.to_string();
        trace!("const eval: {:?} ({})", key, instance);
    }

    let cid = key.value;
    let def_id = cid.instance.def.def_id();

    if def_id.is_local() && tcx.typeck_tables_of(def_id).tainted_by_errors {
        return Err(ErrorHandled::Reported);
    }

    let is_static = tcx.is_static(def_id);

    let span = tcx.def_span(cid.instance.def_id());
    let mut ecx = InterpCx::new(
        tcx.at(span),
        key.param_env,
        CompileTimeInterpreter::new(),
        MemoryExtra { can_access_statics: is_static },
    );

    let res = ecx.load_mir(cid.instance.def, cid.promoted);
    res.and_then(|body| eval_body_using_ecx(&mut ecx, cid, *body))
        .and_then(|place| {
            Ok(RawConst { alloc_id: place.ptr.assert_ptr().alloc_id, ty: place.layout.ty })
        })
        .map_err(|error| {
            let err = error_to_const_error(&ecx, error);
            // errors in statics are always emitted as fatal errors
            if is_static {
                // Ensure that if the above error was either `TooGeneric` or `Reported`
                // an error must be reported.
                let v = err.report_as_error(ecx.tcx, "could not evaluate static initializer");

                // If this is `Reveal:All`, then we need to make sure an error is reported but if
                // this is `Reveal::UserFacing`, then it's expected that we could get a
                // `TooGeneric` error. When we fall back to `Reveal::All`, then it will either
                // succeed or we'll report this error then.
                if key.param_env.reveal == Reveal::All {
                    tcx.sess.delay_span_bug(
                        err.span,
                        &format!("static eval failure did not emit an error: {:#?}", v),
                    );
                }

                v
            } else if def_id.is_local() {
                // constant defined in this crate, we can figure out a lint level!
                match tcx.def_kind(def_id) {
                    // constants never produce a hard error at the definition site. Anything else is
                    // a backwards compatibility hazard (and will break old versions of winapi for
                    // sure)
                    //
                    // note that validation may still cause a hard error on this very same constant,
                    // because any code that existed before validation could not have failed
                    // validation thus preventing such a hard error from being a backwards
                    // compatibility hazard
                    Some(DefKind::Const) | Some(DefKind::AssocConst) => {
                        let hir_id = tcx.hir().as_local_hir_id(def_id).unwrap();
                        err.report_as_lint(
                            tcx.at(tcx.def_span(def_id)),
                            "any use of this value will cause an error",
                            hir_id,
                            Some(err.span),
                        )
                    }
                    // promoting runtime code is only allowed to error if it references broken
                    // constants any other kind of error will be reported to the user as a
                    // deny-by-default lint
                    _ => {
                        if let Some(p) = cid.promoted {
                            let span = tcx.promoted_mir(def_id)[p].span;
                            if let err_inval!(ReferencedConstant) = err.error {
                                err.report_as_error(
                                    tcx.at(span),
                                    "evaluation of constant expression failed",
                                )
                            } else {
                                err.report_as_lint(
                                    tcx.at(span),
                                    "reaching this expression at runtime will panic or abort",
                                    tcx.hir().as_local_hir_id(def_id).unwrap(),
                                    Some(err.span),
                                )
                            }
                        // anything else (array lengths, enum initializers, constant patterns) are
                        // reported as hard errors
                        } else {
                            err.report_as_error(ecx.tcx, "evaluation of constant value failed")
                        }
                    }
                }
            } else {
                // use of broken constant from other crate
                err.report_as_error(ecx.tcx, "could not evaluate constant")
            }
        })
}
