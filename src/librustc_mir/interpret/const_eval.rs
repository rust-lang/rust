use rustc::ty::{self, TyCtxt, Ty, Instance};
use rustc::ty::layout::{self, LayoutOf};
use rustc::ty::subst::Substs;
use rustc::hir::def_id::DefId;
use rustc::mir;
use rustc::middle::const_val::ErrKind::{CheckMatchError, TypeckError};
use rustc::middle::const_val::{ConstEvalErr, ConstVal};
use const_eval::lookup_const_by_id;

use syntax::ast::Mutability;
use syntax::codemap::Span;

use rustc::mir::interpret::{EvalResult, EvalError, EvalErrorKind, GlobalId, Value, MemoryPointer, Pointer, PrimVal};
use super::{Place, EvalContext, StackPopCleanup, ValTy, PlaceExtra};

use std::fmt;
use std::error::Error;

pub fn mk_eval_cx<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    instance: Instance<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
) -> EvalResult<'tcx, EvalContext<'a, 'tcx, CompileTimeEvaluator>> {
    debug!("mk_eval_cx: {:?}, {:?}", instance, param_env);
    let limits = super::ResourceLimits::default();
    let mut ecx = EvalContext::new(tcx, param_env, limits, CompileTimeEvaluator, ());
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

pub fn eval_body<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    instance: Instance<'tcx>,
    promoted: Option<mir::Promoted>,
    param_env: ty::ParamEnv<'tcx>,
) -> EvalResult<'tcx, (Value, Pointer, Ty<'tcx>)> {
    eval_body_and_ecx(tcx, instance, promoted, param_env).0
}

pub fn check_body<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    instance: Instance<'tcx>,
    promoted: Option<mir::Promoted>,
    param_env: ty::ParamEnv<'tcx>,
) {
    let (res, ecx) = eval_body_and_ecx(tcx, instance, promoted, param_env);
    if let Err(mut err) = res {
        ecx.report(&mut err);
    }
}

fn eval_body_and_ecx<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    instance: Instance<'tcx>,
    promoted: Option<mir::Promoted>,
    param_env: ty::ParamEnv<'tcx>,
) -> (EvalResult<'tcx, (Value, Pointer, Ty<'tcx>)>, EvalContext<'a, 'tcx, CompileTimeEvaluator>) {
    debug!("eval_body: {:?}, {:?}", instance, param_env);
    let limits = super::ResourceLimits::default();
    let mut ecx = EvalContext::new(tcx, param_env, limits, CompileTimeEvaluator, ());
    let cid = GlobalId {
        instance,
        promoted,
    };

    let res = (|| {
        if ecx.tcx.has_attr(instance.def_id(), "linkage") {
            return Err(ConstEvalError::NotConst("extern global".to_string()).into());
        }
        let instance_ty = instance.ty(tcx);
        if tcx.interpret_interner.borrow().get_cached(cid).is_none() {
            let mir = ecx.load_mir(instance.def)?;
            let layout = ecx.layout_of(instance_ty)?;
            assert!(!layout.is_unsized());
            let ptr = ecx.memory.allocate(
                layout.size.bytes(),
                layout.align,
                None,
            )?;
            tcx.interpret_interner.borrow_mut().cache(cid, ptr.alloc_id);
            let cleanup = StackPopCleanup::MarkStatic(Mutability::Immutable);
            let name = ty::tls::with(|tcx| tcx.item_path_str(instance.def_id()));
            trace!("const_eval: pushing stack frame for global: {}", name);
            ecx.push_stack_frame(
                instance,
                mir.span,
                mir,
                Place::from_ptr(ptr, layout.align),
                cleanup.clone(),
            )?;

            while ecx.step()? {}
        }
        let alloc = tcx.interpret_interner.borrow().get_cached(cid).expect("global not cached");
        let align = ecx.layout_of(instance_ty)?.align;
        let ptr = MemoryPointer::new(alloc, 0).into();
        let value = match ecx.try_read_value(ptr, align, instance_ty)? {
            Some(val) => val,
            _ => Value::ByRef(ptr, align),
        };
        Ok((value, ptr, instance_ty))
    })();
    (res, ecx)
}

pub fn eval_body_as_integer<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    instance: Instance<'tcx>,
    promoted: Option<mir::Promoted>,
) -> EvalResult<'tcx, u128> {
    let (value, _, ty) = eval_body(tcx, instance, promoted, param_env)?;
    match value {
        Value::ByVal(prim) => prim.to_bytes(),
        _ => err!(TypeNotPrimitive(ty)),
    }
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
            NotConst(ref msg) => write!(f, "Cannot evaluate within constants: \"{}\"", msg),
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

impl<'tcx> super::Machine<'tcx> for CompileTimeEvaluator {
    type MemoryData = ();
    type MemoryKinds = !;
    fn eval_fn_call<'a>(
        ecx: &mut EvalContext<'a, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        destination: Option<(Place, mir::BasicBlock)>,
        _args: &[ValTy<'tcx>],
        span: Span,
        _sig: ty::FnSig<'tcx>,
    ) -> EvalResult<'tcx, bool> {
        debug!("eval_fn_call: {:?}", instance);
        if !ecx.tcx.is_const_fn(instance.def_id()) {
            return Err(
                ConstEvalError::NotConst(format!("calling non-const fn `{}`", instance)).into(),
            );
        }
        let mir = match ecx.load_mir(instance.def) {
            Ok(mir) => mir,
            Err(EvalError { kind: EvalErrorKind::NoMirFor(path), .. }) => {
                return Err(
                    ConstEvalError::NeedsRfc(format!("calling extern function `{}`", path))
                        .into(),
                );
            }
            Err(other) => return Err(other),
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
        ecx: &mut EvalContext<'a, 'tcx, Self>,
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
        _ecx: &EvalContext<'a, 'tcx, Self>,
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

    fn mark_static_initialized(m: !) -> EvalResult<'tcx> {
        m
    }

    fn box_alloc<'a>(
        _ecx: &mut EvalContext<'a, 'tcx, Self>,
        _ty: Ty<'tcx>,
        _dest: Place,
    ) -> EvalResult<'tcx> {
        Err(
            ConstEvalError::NeedsRfc("Heap allocations via `box` keyword".to_string()).into(),
        )
    }

    fn global_item_with_linkage<'a>(
        _ecx: &mut EvalContext<'a, 'tcx, Self>,
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
    val: Value,
    ty: Ty<'tcx>,
) -> ::rustc::middle::const_val::EvalResult<'tcx> {
    match const_val_field_inner(tcx, param_env, instance, variant, field, val, ty) {
        Ok((field, ty)) => Ok(tcx.mk_const(ty::Const {
            val: ConstVal::Value(field),
            ty,
        })),
        Err(err) => Err(ConstEvalErr {
            span: tcx.def_span(instance.def_id()),
            kind: err.into(),
        }),
    }
}

fn const_val_field_inner<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    instance: ty::Instance<'tcx>,
    variant: Option<usize>,
    field: mir::Field,
    value: Value,
    ty: Ty<'tcx>,
) -> ::rustc::mir::interpret::EvalResult<'tcx, (Value, Ty<'tcx>)> {
    trace!("const_val_field: {:?}, {:?}, {:?}, {:?}", instance, field, value, ty);
    let mut ecx = mk_eval_cx(tcx, instance, param_env).unwrap();
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
    key: ty::ParamEnvAnd<'tcx, (DefId, &'tcx Substs<'tcx>)>,
) -> ::rustc::middle::const_val::EvalResult<'tcx> {
    trace!("const eval: {:?}", key);
    let (def_id, substs) = if let Some(resolved) = lookup_const_by_id(tcx, key) {
        resolved
    } else {
        return Err(ConstEvalErr {
            span: tcx.def_span(key.value.0),
            kind: TypeckError
        });
    };

    let tables = tcx.typeck_tables_of(def_id);
    let body = if let Some(id) = tcx.hir.as_local_node_id(def_id) {
        let body_id = tcx.hir.body_owned_by(id);

        // Do match-check before building MIR
        if tcx.check_match(def_id).is_err() {
            return Err(ConstEvalErr {
                span: tcx.def_span(key.value.0),
                kind: CheckMatchError,
            });
        }

        tcx.mir_const_qualif(def_id);
        tcx.hir.body(body_id)
    } else {
        tcx.extern_const_body(def_id).body
    };

    // do not continue into miri if typeck errors occurred
    // it will fail horribly
    if tables.tainted_by_errors {
        return Err(ConstEvalErr { span: body.value.span, kind: TypeckError })
    }


    let instance = ty::Instance::new(def_id, substs);
    match ::interpret::eval_body(tcx, instance, None, key.param_env) {
        Ok((miri_value, _, miri_ty)) => Ok(tcx.mk_const(ty::Const {
            val: ConstVal::Value(miri_value),
            ty: miri_ty,
        })),
        Err(err) => {
            Err(ConstEvalErr { span: body.value.span, kind: err.into() })
        }
    }
}
