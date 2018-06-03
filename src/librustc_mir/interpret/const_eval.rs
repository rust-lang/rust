use rustc::hir;
use rustc::middle::const_val::{ConstEvalErr, ErrKind};
use rustc::middle::const_val::ErrKind::{TypeckError, CheckMatchError};
use rustc::mir;
use rustc::ty::{self, TyCtxt, Ty, Instance};
use rustc::ty::layout::{self, LayoutOf, Primitive};
use rustc::ty::subst::Subst;

use syntax::ast::Mutability;
use syntax::codemap::Span;
use syntax::codemap::DUMMY_SP;

use rustc::mir::interpret::{
    EvalResult, EvalError, EvalErrorKind, GlobalId,
    Value, Scalar, AllocId, Allocation, ConstValue,
};
use super::{Place, EvalContext, StackPopCleanup, ValTy, PlaceExtra, Memory, MemoryKind};

use std::fmt;
use std::error::Error;
use rustc_data_structures::sync::Lrc;

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

pub fn eval_promoted<'a, 'mir, 'tcx>(
    ecx: &mut EvalContext<'a, 'mir, 'tcx, CompileTimeEvaluator>,
    cid: GlobalId<'tcx>,
    mir: &'mir mir::Mir<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
) -> EvalResult<'tcx, (Value, Scalar, Ty<'tcx>)> {
    ecx.with_fresh_body(|ecx| {
        eval_body_using_ecx(ecx, cid, Some(mir), param_env)
    })
}

pub fn value_to_const_value<'tcx>(
    ecx: &EvalContext<'_, '_, 'tcx, CompileTimeEvaluator>,
    val: Value,
    ty: Ty<'tcx>,
) -> &'tcx ty::Const<'tcx> {
    let layout = ecx.tcx.layout_of(ty::ParamEnv::reveal_all().and(ty)).unwrap();
    match (val, &layout.abi) {
        (Value::Scalar(Scalar::Bits { defined: 0, ..}), _) if layout.is_zst() => {},
        (Value::ByRef(..), _) |
        (Value::Scalar(_), &layout::Abi::Scalar(_)) |
        (Value::ScalarPair(..), &layout::Abi::ScalarPair(..)) => {},
        _ => bug!("bad value/layout combo: {:#?}, {:#?}", val, layout),
    }
    let val = (|| {
        match val {
            Value::Scalar(val) => Ok(ConstValue::Scalar(val)),
            Value::ScalarPair(a, b) => Ok(ConstValue::ScalarPair(a, b)),
            Value::ByRef(ptr, align) => {
                let ptr = ptr.to_ptr().unwrap();
                let alloc = ecx.memory.get(ptr.alloc_id)?;
                assert!(alloc.align.abi() >= align.abi());
                assert!(alloc.bytes.len() as u64 - ptr.offset.bytes() >= layout.size.bytes());
                let mut alloc = alloc.clone();
                alloc.align = align;
                let alloc = ecx.tcx.intern_const_alloc(alloc);
                Ok(ConstValue::ByRef(alloc, ptr.offset))
            }
        }
    })();
    match val {
        Ok(val) => ty::Const::from_const_value(ecx.tcx.tcx, val, ty),
        Err(err) => {
            let (frames, span) = ecx.generate_stacktrace(None);
            let err = ConstEvalErr {
                span,
                kind: ErrKind::Miri(err, frames).into(),
            };
            err.report_as_error(
                ecx.tcx,
                "failed to convert Value to ConstValue, this is a bug",
            );
            span_bug!(span, "miri error occured when converting Value to ConstValue")
        }
    }
}

fn eval_body_and_ecx<'a, 'mir, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    cid: GlobalId<'tcx>,
    mir: Option<&'mir mir::Mir<'tcx>>,
    param_env: ty::ParamEnv<'tcx>,
) -> (EvalResult<'tcx, (Value, Scalar, Ty<'tcx>)>, EvalContext<'a, 'mir, 'tcx, CompileTimeEvaluator>) {
    debug!("eval_body_and_ecx: {:?}, {:?}", cid, param_env);
    // we start out with the best span we have
    // and try improving it down the road when more information is available
    let span = tcx.def_span(cid.instance.def_id());
    let span = mir.map(|mir| mir.span).unwrap_or(span);
    let mut ecx = EvalContext::new(tcx.at(span), param_env, CompileTimeEvaluator, ());
    let r = eval_body_using_ecx(&mut ecx, cid, mir, param_env);
    (r, ecx)
}

fn eval_body_using_ecx<'a, 'mir, 'tcx>(
    ecx: &mut EvalContext<'a, 'mir, 'tcx, CompileTimeEvaluator>,
    cid: GlobalId<'tcx>,
    mir: Option<&'mir mir::Mir<'tcx>>,
    param_env: ty::ParamEnv<'tcx>,
) -> EvalResult<'tcx, (Value, Scalar, Ty<'tcx>)> {
    debug!("eval_body: {:?}, {:?}", cid, param_env);
    let tcx = ecx.tcx.tcx;
    let mut mir = match mir {
        Some(mir) => mir,
        None => ecx.load_mir(cid.instance.def)?,
    };
    if let Some(index) = cid.promoted {
        mir = &mir.promoted[index];
    }
    let layout = ecx.layout_of(mir.return_ty().subst(tcx, cid.instance.substs))?;
    assert!(!layout.is_unsized());
    let ptr = ecx.memory.allocate(
        layout.size,
        layout.align,
        None,
    )?;
    let internally_mutable = !layout.ty.is_freeze(tcx, param_env, mir.span);
    let is_static = tcx.is_static(cid.instance.def_id());
    let mutability = if is_static == Some(hir::Mutability::MutMutable) || internally_mutable {
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
    let ptr = ptr.into();
    // always try to read the value and report errors
    let value = match ecx.try_read_value(ptr, layout.align, layout.ty)? {
        Some(val) if is_static.is_none() => val,
        // point at the allocation
        _ => Value::ByRef(ptr, layout.align),
    };
    Ok((value, ptr, layout.ty))
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
        args: &[ValTy<'tcx>],
        dest: Place,
        dest_layout: layout::TyLayout<'tcx>,
        target: mir::BasicBlock,
    ) -> EvalResult<'tcx> {
        let substs = instance.substs;

        let intrinsic_name = &ecx.tcx.item_name(instance.def_id()).as_str()[..];
        match intrinsic_name {
            "min_align_of" => {
                let elem_ty = substs.type_at(0);
                let elem_align = ecx.layout_of(elem_ty)?.align.abi();
                let align_val = Scalar::Bits {
                    bits: elem_align as u128,
                    defined: dest_layout.size.bits() as u8,
                };
                ecx.write_scalar(dest, align_val, dest_layout.ty)?;
            }

            "size_of" => {
                let ty = substs.type_at(0);
                let size = ecx.layout_of(ty)?.size.bytes() as u128;
                let size_val = Scalar::Bits {
                    bits: size,
                    defined: dest_layout.size.bits() as u8,
                };
                ecx.write_scalar(dest, size_val, dest_layout.ty)?;
            }

            "type_id" => {
                let ty = substs.type_at(0);
                let type_id = ecx.tcx.type_id_hash(ty) as u128;
                let id_val = Scalar::Bits {
                    bits: type_id,
                    defined: dest_layout.size.bits() as u8,
                };
                ecx.write_scalar(dest, id_val, dest_layout.ty)?;
            }
            "ctpop" | "cttz" | "cttz_nonzero" | "ctlz" | "ctlz_nonzero" | "bswap" => {
                let ty = substs.type_at(0);
                let layout_of = ecx.layout_of(ty)?;
                let bits = ecx.value_to_scalar(args[0])?.to_bits(layout_of.size)?;
                let kind = match layout_of.abi {
                    ty::layout::Abi::Scalar(ref scalar) => scalar.value,
                    _ => Err(::rustc::mir::interpret::EvalErrorKind::TypeNotPrimitive(ty))?,
                };
                let out_val = if intrinsic_name.ends_with("_nonzero") {
                    if bits == 0 {
                        return err!(Intrinsic(format!("{} called on 0", intrinsic_name)));
                    }
                    numeric_intrinsic(intrinsic_name.trim_right_matches("_nonzero"), bits, kind)?
                } else {
                    numeric_intrinsic(intrinsic_name, bits, kind)?
                };
                ecx.write_scalar(dest, out_val, ty)?;
            }

            name => return Err(
                ConstEvalError::NeedsRfc(format!("calling intrinsic `{}`", name)).into()
            ),
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
        left: Scalar,
        _left_ty: Ty<'tcx>,
        right: Scalar,
        _right_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx, Option<(Scalar, bool)>> {
        if left.is_bits() && right.is_bits() {
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
        Ok(ecx
            .tcx
            .alloc_map
            .lock()
            .intern_static(cid.instance.def_id()))
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
    value: ConstValue<'tcx>,
    ty: Ty<'tcx>,
) -> ::rustc::middle::const_val::EvalResult<'tcx> {
    trace!("const_val_field: {:?}, {:?}, {:?}, {:?}", instance, field, value, ty);
    let mut ecx = mk_eval_cx(tcx, instance, param_env).unwrap();
    let result = (|| {
        let value = ecx.const_value_to_value(value, ty)?;
        let layout = ecx.layout_of(ty)?;
        let (ptr, align) = match value {
            Value::ByRef(ptr, align) => (ptr, align),
            Value::ScalarPair(..) | Value::Scalar(_) => {
                let ptr = ecx.alloc_ptr(ty)?.into();
                ecx.write_value_to_ptr(value, ptr, layout.align, ty)?;
                (ptr, layout.align)
            },
        };
        let place = Place::Ptr {
            ptr,
            align,
            extra: variant.map_or(PlaceExtra::None, PlaceExtra::DowncastVariant),
        };
        let (place, layout) = ecx.place_field(place, field, layout)?;
        let (ptr, align) = place.to_ptr_align();
        let mut new_value = Value::ByRef(ptr, align);
        new_value = ecx.try_read_by_ref(new_value, layout.ty)?;
        use rustc_data_structures::indexed_vec::Idx;
        match (value, new_value) {
            (Value::Scalar(_), Value::ByRef(..)) |
            (Value::ScalarPair(..), Value::ByRef(..)) |
            (Value::Scalar(_), Value::ScalarPair(..)) => bug!(
                "field {} of {:?} yielded {:?}",
                field.index(),
                value,
                new_value,
            ),
            _ => {},
        }
        Ok(value_to_const_value(&ecx, new_value, layout.ty))
    })();
    result.map_err(|err| {
        let (trace, span) = ecx.generate_stacktrace(None);
        let err = ErrKind::Miri(err, trace);
        ConstEvalErr {
            kind: err.into(),
            span,
        }
    })
}

pub fn const_variant_index<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    instance: ty::Instance<'tcx>,
    val: ConstValue<'tcx>,
    ty: Ty<'tcx>,
) -> EvalResult<'tcx, usize> {
    trace!("const_variant_index: {:?}, {:?}, {:?}", instance, val, ty);
    let mut ecx = mk_eval_cx(tcx, instance, param_env).unwrap();
    let value = ecx.const_value_to_value(val, ty)?;
    let (ptr, align) = match value {
        Value::ScalarPair(..) | Value::Scalar(_) => {
            let layout = ecx.layout_of(ty)?;
            let ptr = ecx.memory.allocate(layout.size, layout.align, Some(MemoryKind::Stack))?.into();
            ecx.write_value_to_ptr(value, ptr, layout.align, ty)?;
            (ptr, layout.align)
        },
        Value::ByRef(ptr, align) => (ptr, align),
    };
    let place = Place::from_scalar_ptr(ptr, align);
    ecx.read_discriminant_as_variant_index(place, ty)
}

pub fn const_value_to_allocation_provider<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    (val, ty): (ConstValue<'tcx>, Ty<'tcx>),
) -> &'tcx Allocation {
    match val {
        ConstValue::ByRef(alloc, offset) => {
            assert_eq!(offset.bytes(), 0);
            return alloc;
        },
        _ => ()
    }
    let result = || -> EvalResult<'tcx, &'tcx Allocation> {
        let mut ecx = EvalContext::new(
            tcx.at(DUMMY_SP),
            ty::ParamEnv::reveal_all(),
            CompileTimeEvaluator,
            ());
        let value = ecx.const_value_to_value(val, ty)?;
        let layout = ecx.layout_of(ty)?;
        let ptr = ecx.memory.allocate(layout.size, layout.align, Some(MemoryKind::Stack))?;
        ecx.write_value_to_ptr(value, ptr.into(), layout.align, ty)?;
        let alloc = ecx.memory.get(ptr.alloc_id)?;
        Ok(tcx.intern_const_alloc(alloc.clone()))
    };
    result().expect("unable to convert ConstVal to Allocation")
}

pub fn const_eval_provider<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    key: ty::ParamEnvAnd<'tcx, GlobalId<'tcx>>,
) -> ::rustc::middle::const_val::EvalResult<'tcx> {
    trace!("const eval: {:?}", key);
    let cid = key.value;
    let def_id = cid.instance.def.def_id();

    if let Some(id) = tcx.hir.as_local_node_id(def_id) {
        let tables = tcx.typeck_tables_of(def_id);
        let span = tcx.def_span(def_id);

        // Do match-check before building MIR
        if tcx.check_match(def_id).is_err() {
            return Err(ConstEvalErr {
                kind: Lrc::new(CheckMatchError),
                span,
            });
        }

        if let hir::BodyOwnerKind::Const = tcx.hir.body_owner_kind(id) {
            tcx.mir_const_qualif(def_id);
        }

        // Do not continue into miri if typeck errors occurred; it will fail horribly
        if tables.tainted_by_errors {
            return Err(ConstEvalErr {
                kind: Lrc::new(TypeckError),
                span,
            });
        }
    };

    let (res, ecx) = eval_body_and_ecx(tcx, cid, None, key.param_env);
    res.and_then(|(mut val, _, miri_ty)| {
        if tcx.is_static(def_id).is_none() {
            val = ecx.try_read_by_ref(val, miri_ty)?;
        }
        Ok(value_to_const_value(&ecx, val, miri_ty))
    }).map_err(|err| {
        let (trace, span) = ecx.generate_stacktrace(None);
        let err = ErrKind::Miri(err, trace);
        let err = ConstEvalErr {
            kind: err.into(),
            span,
        };
        if tcx.is_static(def_id).is_some() {
            err.report_as_error(ecx.tcx, "could not evaluate static initializer");
        }
        err
    })
}

fn numeric_intrinsic<'tcx>(
    name: &str,
    bits: u128,
    kind: Primitive,
) -> EvalResult<'tcx, Scalar> {
    let defined = match kind {
        Primitive::Int(integer, _) => integer.size().bits() as u8,
        _ => bug!("invalid `{}` argument: {:?}", name, bits),
    };
    let extra = 128 - defined as u128;
    let bits_out = match name {
        "ctpop" => bits.count_ones() as u128,
        "ctlz" => bits.leading_zeros() as u128 - extra,
        "cttz" => (bits << extra).trailing_zeros() as u128 - extra,
        "bswap" => (bits << extra).swap_bytes(),
        _ => bug!("not a numeric intrinsic: {}", name),
    };
    Ok(Scalar::Bits { bits: bits_out, defined })
}
