use rustc::hir;
use rustc::middle::const_val::{ConstEvalErr, ConstVal, ErrKind};
use rustc::middle::const_val::ErrKind::{TypeckError, CheckMatchError};
use rustc::mir;
use rustc::ty::{self, TyCtxt, Ty, Instance};
use rustc::ty::layout::{self, LayoutOf};
use rustc::ty::subst::Subst;

use syntax::ast::Mutability;
use syntax::codemap::Span;

use rustc::mir::interpret::{EvalResult, EvalError, EvalErrorKind, GlobalId, Value, MemoryPointer, Pointer, PrimVal, AllocId};
use super::{Place, EvalContext, StackPopCleanup, ValTy, PlaceExtra, Memory};

use std::fmt;
use std::error::Error;
use std::rc::Rc;

pub fn mk_borrowck_eval_cx<'a, 'mir, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    instance: Instance<'tcx>,
    mir: &'mir mir::Mir<'tcx>,
    span: Span,
) -> EvalResult<'tcx, EvalContext<'a, 'mir, 'tcx, CompileTimeEvaluator>> {
    debug!("mk_borrowck_eval_cx: {:?}", instance);
    let param_env = tcx.param_env(instance.def_id());
    let mut ecx = EvalContext::new(tcx.at(span), param_env, CompileTimeEvaluator, ());
    // insert a stack frame so any queries have the correct substs
    ecx.push_stack_frame(
        instance,
        span,
        mir,
        Place::undef(),
        StackPopCleanup::None,
    )?;
    Ok(ecx)
}

pub fn mk_eval_cx<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    instance: Instance<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
) -> EvalResult<'tcx, EvalContext<'a, 'tcx, 'tcx, CompileTimeEvaluator>> {
    debug!("mk_eval_cx: {:?}, {:?}", instance, param_env);
    let span = tcx.def_span(instance.def_id());
    let mut ecx = EvalContext::new(tcx.at(span), param_env, CompileTimeEvaluator, ());
    let mir = ecx.load_mir(instance.def)?;
    // insert a stack frame so any queries have the correct substs
    ecx.push_stack_frame(
        instance,
        mir.span,
        mir,
        Place::undef(),
        StackPopCleanup::None,
    )?;
    Ok(ecx)
}

pub fn eval_body_with_mir<'a, 'mir, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    cid: GlobalId<'tcx>,
    mir: &'mir mir::Mir<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
) -> Option<(Value, Pointer, Ty<'tcx>)> {
    let (res, ecx) = eval_body_and_ecx(tcx, cid, Some(mir), param_env);
    match res {
        Ok(val) => Some(val),
        Err(mut err) => {
            ecx.report(&mut err, true, None);
            None
        }
    }
}

pub fn eval_body<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    cid: GlobalId<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
) -> Option<(Value, Pointer, Ty<'tcx>)> {
    let (res, ecx) = eval_body_and_ecx(tcx, cid, None, param_env);
    match res {
        Ok(val) => Some(val),
        Err(mut err) => {
            ecx.report(&mut err, true, None);
            None
        }
    }
}

fn eval_body_and_ecx<'a, 'mir, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    cid: GlobalId<'tcx>,
    mir: Option<&'mir mir::Mir<'tcx>>,
    param_env: ty::ParamEnv<'tcx>,
) -> (EvalResult<'tcx, (Value, Pointer, Ty<'tcx>)>, EvalContext<'a, 'mir, 'tcx, CompileTimeEvaluator>) {
    debug!("eval_body: {:?}, {:?}", cid, param_env);
    // we start out with the best span we have
    // and try improving it down the road when more information is available
    let span = tcx.def_span(cid.instance.def_id());
    let mut span = mir.map(|mir| mir.span).unwrap_or(span);
    let mut ecx = EvalContext::new(tcx.at(span), param_env, CompileTimeEvaluator, ());
    let res = (|| {
        let mut mir = match mir {
            Some(mir) => mir,
            None => ecx.load_mir(cid.instance.def)?,
        };
        if let Some(index) = cid.promoted {
            mir = &mir.promoted[index];
        }
        span = mir.span;
        let layout = ecx.layout_of(mir.return_ty().subst(tcx, cid.instance.substs))?;
        let alloc = tcx.interpret_interner.get_cached(cid.instance.def_id());
        let is_static = tcx.is_static(cid.instance.def_id()).is_some();
        let alloc = match alloc {
            Some(alloc) => {
                assert!(cid.promoted.is_none());
                assert!(param_env.caller_bounds.is_empty());
                alloc
            },
            None => {
                assert!(!layout.is_unsized());
                let ptr = ecx.memory.allocate(
                    layout.size.bytes(),
                    layout.align,
                    None,
                )?;
                if is_static {
                    tcx.interpret_interner.cache(cid.instance.def_id(), ptr.alloc_id);
                }
                let internally_mutable = !layout.ty.is_freeze(tcx, param_env, mir.span);
                let mutability = tcx.is_static(cid.instance.def_id());
                let mutability = if mutability == Some(hir::Mutability::MutMutable) || internally_mutable {
                    Mutability::Mutable
                } else {
                    Mutability::Immutable
                };
                let cleanup = StackPopCleanup::MarkStatic(mutability);
                let name = ty::tls::with(|tcx| tcx.item_path_str(cid.instance.def_id()));
                let prom = cid.promoted.map_or(String::new(), |p| format!("::promoted[{:?}]", p));
                trace!("const_eval: pushing stack frame for global: {}{}", name, prom);
                assert!(mir.arg_count == 0);
                ecx.push_stack_frame(
                    cid.instance,
                    mir.span,
                    mir,
                    Place::from_ptr(ptr, layout.align),
                    cleanup,
                )?;

                while ecx.step()? {}
                ptr.alloc_id
            }
        };
        let ptr = MemoryPointer::new(alloc, 0).into();
        // always try to read the value and report errors
        let value = match ecx.try_read_value(ptr, layout.align, layout.ty)? {
            // if it's a constant (so it needs no address, directly compute its value)
            Some(val) if !is_static => val,
            // point at the allocation
            _ => Value::ByRef(ptr, layout.align),
        };
        Ok((value, ptr, layout.ty))
    })();
    (res, ecx)
}

pub struct CompileTimeEvaluator;

impl<'tcx> Into<EvalError<'tcx>> for ConstEvalError {
    fn into(self) -> EvalError<'tcx> {
        EvalErrorKind::MachineError(self.to_string()).into()
    }
}

#[derive(Clone, Debug)]
enum ConstEvalError {
    NeedsRfc(String),
    NotConst(String),
}

impl fmt::Display for ConstEvalError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::ConstEvalError::*;
        match *self {
            NeedsRfc(ref msg) => {
                write!(
                    f,
                    "\"{}\" needs an rfc before being allowed inside constants",
                    msg
                )
            }
            NotConst(ref msg) => write!(f, "{}", msg),
        }
    }
}

impl Error for ConstEvalError {
    fn description(&self) -> &str {
        use self::ConstEvalError::*;
        match *self {
            NeedsRfc(_) => "this feature needs an rfc before being allowed inside constants",
            NotConst(_) => "this feature is not compatible with constant evaluation",
        }
    }

    fn cause(&self) -> Option<&dyn Error> {
        None
    }
}

impl<'mir, 'tcx> super::Machine<'mir, 'tcx> for CompileTimeEvaluator {
    type MemoryData = ();
    type MemoryKinds = !;
    fn eval_fn_call<'a>(
        ecx: &mut EvalContext<'a, 'mir, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        destination: Option<(Place, mir::BasicBlock)>,
        args: &[ValTy<'tcx>],
        span: Span,
        sig: ty::FnSig<'tcx>,
    ) -> EvalResult<'tcx, bool> {
        debug!("eval_fn_call: {:?}", instance);
        if !ecx.tcx.is_const_fn(instance.def_id()) {
            let def_id = instance.def_id();
            let (op, oflo) = if let Some(op) = ecx.tcx.is_binop_lang_item(def_id) {
                op
            } else {
                return Err(
                    ConstEvalError::NotConst(format!("calling non-const fn `{}`", instance)).into(),
                );
            };
            let (dest, bb) = destination.expect("128 lowerings can't diverge");
            let dest_ty = sig.output();
            if oflo {
                ecx.intrinsic_with_overflow(op, args[0], args[1], dest, dest_ty)?;
            } else {
                ecx.intrinsic_overflowing(op, args[0], args[1], dest, dest_ty)?;
            }
            ecx.goto_block(bb);
            return Ok(true);
        }
        let mir = match ecx.load_mir(instance.def) {
            Ok(mir) => mir,
            Err(err) => {
                if let EvalErrorKind::NoMirFor(ref path) = err.kind {
                    return Err(
                        ConstEvalError::NeedsRfc(format!("calling extern function `{}`", path))
                            .into(),
                    );
                }
                return Err(err);
            }
        };
        let (return_place, return_to_block) = match destination {
            Some((place, block)) => (place, StackPopCleanup::Goto(block)),
            None => (Place::undef(), StackPopCleanup::None),
        };

        ecx.push_stack_frame(
            instance,
            span,
            mir,
            return_place,
            return_to_block,
        )?;

        Ok(false)
    }


    fn call_intrinsic<'a>(
        ecx: &mut EvalContext<'a, 'mir, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        _args: &[ValTy<'tcx>],
        dest: Place,
        dest_layout: layout::TyLayout<'tcx>,
        target: mir::BasicBlock,
    ) -> EvalResult<'tcx> {
        let substs = instance.substs;

        let intrinsic_name = &ecx.tcx.item_name(instance.def_id())[..];
        match intrinsic_name {
            "min_align_of" => {
                let elem_ty = substs.type_at(0);
                let elem_align = ecx.layout_of(elem_ty)?.align.abi();
                let align_val = PrimVal::from_u128(elem_align as u128);
                ecx.write_primval(dest, align_val, dest_layout.ty)?;
            }

            "size_of" => {
                let ty = substs.type_at(0);
                let size = ecx.layout_of(ty)?.size.bytes() as u128;
                ecx.write_primval(dest, PrimVal::from_u128(size), dest_layout.ty)?;
            }

            "type_id" => {
                let ty = substs.type_at(0);
                let type_id = ecx.tcx.type_id_hash(ty) as u128;
                ecx.write_primval(dest, PrimVal::from_u128(type_id), dest_layout.ty)?;
            }

            name => return Err(ConstEvalError::NeedsRfc(format!("calling intrinsic `{}`", name)).into()),
        }

        ecx.goto_block(target);

        // Since we pushed no stack frame, the main loop will act
        // as if the call just completed and it's returning to the
        // current frame.
        Ok(())
    }

    fn try_ptr_op<'a>(
        _ecx: &EvalContext<'a, 'mir, 'tcx, Self>,
        _bin_op: mir::BinOp,
        left: PrimVal,
        _left_ty: Ty<'tcx>,
        right: PrimVal,
        _right_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx, Option<(PrimVal, bool)>> {
        if left.is_bytes() && right.is_bytes() {
            Ok(None)
        } else {
            Err(
                ConstEvalError::NeedsRfc("Pointer arithmetic or comparison".to_string()).into(),
            )
        }
    }

    fn mark_static_initialized<'a>(
        _mem: &mut Memory<'a, 'mir, 'tcx, Self>,
        _id: AllocId,
        _mutability: Mutability,
    ) -> EvalResult<'tcx, bool> {
        Ok(false)
    }

    fn init_static<'a>(
        ecx: &mut EvalContext<'a, 'mir, 'tcx, Self>,
        cid: GlobalId<'tcx>,
    ) -> EvalResult<'tcx, AllocId> {
        let alloc = ecx
                    .tcx
                    .interpret_interner
                    .get_cached(cid.instance.def_id());
        // Don't evaluate when already cached to prevent cycles
        if let Some(alloc) = alloc {
            return Ok(alloc)
        }
        // ensure the static is computed
        ecx.const_eval(cid)?;
        Ok(ecx
            .tcx
            .interpret_interner
            .get_cached(cid.instance.def_id())
            .expect("uncached static"))
    }

    fn box_alloc<'a>(
        _ecx: &mut EvalContext<'a, 'mir, 'tcx, Self>,
        _ty: Ty<'tcx>,
        _dest: Place,
    ) -> EvalResult<'tcx> {
        Err(
            ConstEvalError::NeedsRfc("Heap allocations via `box` keyword".to_string()).into(),
        )
    }

    fn global_item_with_linkage<'a>(
        _ecx: &mut EvalContext<'a, 'mir, 'tcx, Self>,
        _instance: ty::Instance<'tcx>,
        _mutability: Mutability,
    ) -> EvalResult<'tcx> {
        Err(
            ConstEvalError::NotConst("statics with `linkage` attribute".to_string()).into(),
        )
    }
}

pub fn const_val_field<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    instance: ty::Instance<'tcx>,
    variant: Option<usize>,
    field: mir::Field,
    value: Value,
    ty: Ty<'tcx>,
) -> ::rustc::middle::const_val::EvalResult<'tcx> {
    trace!("const_val_field: {:?}, {:?}, {:?}, {:?}", instance, field, value, ty);
    let mut ecx = mk_eval_cx(tcx, instance, param_env).unwrap();
    let result = (|| {
        let (mut field, ty) = match value {
            Value::ByValPair(..) | Value::ByVal(_) => ecx.read_field(value, variant, field, ty)?.expect("const_val_field on non-field"),
            Value::ByRef(ptr, align) => {
                let place = Place::Ptr {
                    ptr,
                    align,
                    extra: variant.map_or(PlaceExtra::None, PlaceExtra::DowncastVariant),
                };
                let layout = ecx.layout_of(ty)?;
                let (place, layout) = ecx.place_field(place, field, layout)?;
                let (ptr, align) = place.to_ptr_align();
                (Value::ByRef(ptr, align), layout.ty)
            }
        };
        if let Value::ByRef(ptr, align) = field {
            if let Some(val) = ecx.try_read_value(ptr, align, ty)? {
                field = val;
            }
        }
        Ok((field, ty))
    })();
    match result {
        Ok((field, ty)) => Ok(tcx.mk_const(ty::Const {
            val: ConstVal::Value(field),
            ty,
        })),
        Err(err) => {
            let (trace, span) = ecx.generate_stacktrace(None);
            let err = ErrKind::Miri(err, trace);
            Err(ConstEvalErr {
                kind: err.into(),
                span,
            })
        },
    }
}

pub fn const_discr<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    instance: ty::Instance<'tcx>,
    value: Value,
    ty: Ty<'tcx>,
) -> EvalResult<'tcx, u128> {
    trace!("const_discr: {:?}, {:?}, {:?}", instance, value, ty);
    let mut ecx = mk_eval_cx(tcx, instance, param_env).unwrap();
    let (ptr, align) = match value {
        Value::ByValPair(..) | Value::ByVal(_) => {
            let layout = ecx.layout_of(ty)?;
            use super::MemoryKind;
            let ptr = ecx.memory.allocate(layout.size.bytes(), layout.align, Some(MemoryKind::Stack))?;
            let ptr: Pointer = ptr.into();
            ecx.write_value_to_ptr(value, ptr, layout.align, ty)?;
            (ptr, layout.align)
        },
        Value::ByRef(ptr, align) => (ptr, align),
    };
    let place = Place::from_primval_ptr(ptr, align);
    ecx.read_discriminant_value(place, ty)
}

pub fn const_eval_provider<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    key: ty::ParamEnvAnd<'tcx, GlobalId<'tcx>>,
) -> ::rustc::middle::const_val::EvalResult<'tcx> {
    trace!("const eval: {:?}", key);
    let cid = key.value;
    let def_id = cid.instance.def.def_id();

    if tcx.is_foreign_item(def_id) {
        let id = tcx.interpret_interner.get_cached(def_id);
        let id = match id {
            // FIXME: due to caches this shouldn't happen, add some assertions
            Some(id) => id,
            None => {
                let id = tcx.interpret_interner.reserve();
                tcx.interpret_interner.cache(def_id, id);
                id
            },
        };
        let ty = tcx.type_of(def_id);
        let layout = tcx.layout_of(key.param_env.and(ty)).unwrap();
        let ptr = MemoryPointer::new(id, 0);
        return Ok(tcx.mk_const(ty::Const {
            val: ConstVal::Value(Value::ByRef(ptr.into(), layout.align)),
            ty,
        }))
    }

    if let Some(id) = tcx.hir.as_local_node_id(def_id) {
        let tables = tcx.typeck_tables_of(def_id);
        let span = tcx.def_span(def_id);

        // Do match-check before building MIR
        if tcx.check_match(def_id).is_err() {
            return Err(ConstEvalErr {
                kind: Rc::new(CheckMatchError),
                span,
            });
        }

        if let hir::BodyOwnerKind::Const = tcx.hir.body_owner_kind(id) {
            tcx.mir_const_qualif(def_id);
        }

        // Do not continue into miri if typeck errors occurred; it will fail horribly
        if tables.tainted_by_errors {
            return Err(ConstEvalErr {
                kind: Rc::new(TypeckError),
                span,
            });
        }
    };

    let (res, ecx) = eval_body_and_ecx(tcx, cid, None, key.param_env);
    res.map(|(miri_value, _, miri_ty)| {
        tcx.mk_const(ty::Const {
            val: ConstVal::Value(miri_value),
            ty: miri_ty,
        })
    }).map_err(|mut err| {
        if tcx.is_static(def_id).is_some() {
            ecx.report(&mut err, true, None);
        }
        let (trace, span) = ecx.generate_stacktrace(None);
        let err = ErrKind::Miri(err, trace);
        ConstEvalErr {
            kind: err.into(),
            span,
        }
    })
}
