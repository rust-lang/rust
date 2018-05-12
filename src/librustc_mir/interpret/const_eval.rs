use rustc::hir;
use rustc::middle::const_val::{ConstEvalErr, ConstVal, ErrKind};
use rustc::middle::const_val::ErrKind::{TypeckError, CheckMatchError};
use rustc::mir;
use rustc::ty::{self, TyCtxt, Ty, Instance};
use rustc::ty::layout::{self, LayoutOf};
use rustc::ty::subst::Subst;

use syntax::ast::Mutability;
use syntax::codemap::Span;

use rustc::mir::interpret::{EvalResult, EvalError, EvalErrorKind, GlobalId, Value, MemoryPointer, Pointer, PrimVal, PrimValKind, AllocId};
use super::{Place, EvalContext, StackPopCleanup, ValTy, PlaceExtra, Memory};

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
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    cid: GlobalId<'tcx>,
    mir: &'mir mir::Mir<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
) -> Option<(Value, Pointer, Ty<'tcx>)> {
    let (res, ecx) = eval_body_and_ecx(tcx, cid, Some(mir), param_env);
    match res {
        Ok(val) => Some(val),
        Err(mut err) => {
            ecx.report(&mut err, false, None);
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
        assert!(!layout.is_unsized());
        let ptr = ecx.memory.allocate(
            layout.size.bytes(),
            layout.align,
            None,
        )?;
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
        let ptr = ptr.into();
        // always try to read the value and report errors
        let value = match ecx.try_read_value(ptr, layout.align, layout.ty)? {
            // if it's a constant (so it needs no address, directly compute its value)
            Some(val) if tcx.is_static(cid.instance.def_id()).is_none() => val,
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
        args: &[ValTy<'tcx>],
        dest: Place,
        dest_layout: layout::TyLayout<'tcx>,
        target: mir::BasicBlock,
    ) -> EvalResult<'tcx> {
        let substs = instance.substs;

        let intrinsic_name = &ecx.tcx.item_name(instance.def_id()).as_str()[..];
        match intrinsic_name {
            "add_with_overflow" => {
                ecx.intrinsic_with_overflow(
                    mir::BinOp::Add,
                    args[0],
                    args[1],
                    dest,
                    dest_layout.ty,
                )?
            }

            "sub_with_overflow" => {
                ecx.intrinsic_with_overflow(
                    mir::BinOp::Sub,
                    args[0],
                    args[1],
                    dest,
                    dest_layout.ty,
                )?
            }

            "mul_with_overflow" => {
                ecx.intrinsic_with_overflow(
                    mir::BinOp::Mul,
                    args[0],
                    args[1],
                    dest,
                    dest_layout.ty,
                )?
            }

            "overflowing_sub" => {
                ecx.intrinsic_overflowing(
                    mir::BinOp::Sub,
                    args[0],
                    args[1],
                    dest,
                    dest_layout.ty,
                )?;
            }

            "overflowing_mul" => {
                ecx.intrinsic_overflowing(
                    mir::BinOp::Mul,
                    args[0],
                    args[1],
                    dest,
                    dest_layout.ty,
                )?;
            }

            "overflowing_add" => {
                ecx.intrinsic_overflowing(
                    mir::BinOp::Add,
                    args[0],
                    args[1],
                    dest,
                    dest_layout.ty,
                )?;
            }

            "powf32" => {
                let f = ecx.value_to_primval(args[0])?.to_bytes()?;
                let f = f32::from_bits(f as u32);
                let f2 = ecx.value_to_primval(args[1])?.to_bytes()?;
                let f2 = f32::from_bits(f2 as u32);
                ecx.write_primval(
                    dest,
                    PrimVal::Bytes(f.powf(f2).to_bits() as u128),
                    dest_layout.ty,
                )?;
            }

            "powf64" => {
                let f = ecx.value_to_primval(args[0])?.to_bytes()?;
                let f = f64::from_bits(f as u64);
                let f2 = ecx.value_to_primval(args[1])?.to_bytes()?;
                let f2 = f64::from_bits(f2 as u64);
                ecx.write_primval(
                    dest,
                    PrimVal::Bytes(f.powf(f2).to_bits() as u128),
                    dest_layout.ty,
                )?;
            }

            "fmaf32" => {
                let a = ecx.value_to_primval(args[0])?.to_bytes()?;
                let a = f32::from_bits(a as u32);
                let b = ecx.value_to_primval(args[1])?.to_bytes()?;
                let b = f32::from_bits(b as u32);
                let c = ecx.value_to_primval(args[2])?.to_bytes()?;
                let c = f32::from_bits(c as u32);
                ecx.write_primval(
                    dest,
                    PrimVal::Bytes((a * b + c).to_bits() as u128),
                    dest_layout.ty,
                )?;
            }

            "fmaf64" => {
                let a = ecx.value_to_primval(args[0])?.to_bytes()?;
                let a = f64::from_bits(a as u64);
                let b = ecx.value_to_primval(args[1])?.to_bytes()?;
                let b = f64::from_bits(b as u64);
                let c = ecx.value_to_primval(args[2])?.to_bytes()?;
                let c = f64::from_bits(c as u64);
                ecx.write_primval(
                    dest,
                    PrimVal::Bytes((a * b + c).to_bits() as u128),
                    dest_layout.ty,
                )?;
            }

            "powif32" => {
                let f = ecx.value_to_primval(args[0])?.to_bytes()?;
                let f = f32::from_bits(f as u32);
                let i = ecx.value_to_primval(args[1])?.to_i128()?;
                ecx.write_primval(
                    dest,
                    PrimVal::Bytes(f.powi(i as i32).to_bits() as u128),
                    dest_layout.ty,
                )?;
            }

            "powif64" => {
                let f = ecx.value_to_primval(args[0])?.to_bytes()?;
                let f = f64::from_bits(f as u64);
                let i = ecx.value_to_primval(args[1])?.to_i128()?;
                ecx.write_primval(
                    dest,
                    PrimVal::Bytes(f.powi(i as i32).to_bits() as u128),
                    dest_layout.ty,
                )?;
            }

            "unchecked_shl" => {
                let bits = dest_layout.size.bytes() as u128 * 8;
                let rhs = ecx.value_to_primval(args[1])?
                    .to_bytes()?;
                if rhs >= bits {
                    return err!(Intrinsic(
                        format!("Overflowing shift by {} in unchecked_shl", rhs),
                    ));
                }
                ecx.intrinsic_overflowing(
                    mir::BinOp::Shl,
                    args[0],
                    args[1],
                    dest,
                    dest_layout.ty,
                )?;
            }

            "unchecked_shr" => {
                let bits = dest_layout.size.bytes() as u128 * 8;
                let rhs = ecx.value_to_primval(args[1])?
                    .to_bytes()?;
                if rhs >= bits {
                    return err!(Intrinsic(
                        format!("Overflowing shift by {} in unchecked_shr", rhs),
                    ));
                }
                ecx.intrinsic_overflowing(
                    mir::BinOp::Shr,
                    args[0],
                    args[1],
                    dest,
                    dest_layout.ty,
                )?;
            }

            "unchecked_div" => {
                let rhs = ecx.value_to_primval(args[1])?
                    .to_bytes()?;
                if rhs == 0 {
                    return err!(Intrinsic(format!("Division by 0 in unchecked_div")));
                }
                ecx.intrinsic_overflowing(
                    mir::BinOp::Div,
                    args[0],
                    args[1],
                    dest,
                    dest_layout.ty,
                )?;
            }

            "unchecked_rem" => {
                let rhs = ecx.value_to_primval(args[1])?
                    .to_bytes()?;
                if rhs == 0 {
                    return err!(Intrinsic(format!("Division by 0 in unchecked_rem")));
                }
                ecx.intrinsic_overflowing(
                    mir::BinOp::Rem,
                    args[0],
                    args[1],
                    dest,
                    dest_layout.ty,
                )?;
            }

            "ctpop" | "cttz" | "cttz_nonzero" | "ctlz" | "ctlz_nonzero" | "bswap" => {
                let ty = substs.type_at(0);
                let num = ecx.value_to_primval(args[0])?.to_bytes()?;
                let kind = ecx.ty_to_primval_kind(ty)?;
                let num = if intrinsic_name.ends_with("_nonzero") {
                    if num == 0 {
                        return err!(Intrinsic(format!("{} called on 0", intrinsic_name)));
                    }
                    numeric_intrinsic(intrinsic_name.trim_right_matches("_nonzero"), num, kind)?
                } else {
                    numeric_intrinsic(intrinsic_name, num, kind)?
                };
                ecx.write_primval(dest, num, ty)?;
            }

            "sinf32" | "fabsf32" | "cosf32" | "sqrtf32" | "expf32" | "exp2f32" | "logf32" |
            "log10f32" | "log2f32" | "floorf32" | "ceilf32" | "truncf32" => {
                let f = ecx.value_to_primval(args[0])?.to_bytes()?;
                let f = f32::from_bits(f as u32);
                let f = match intrinsic_name {
                    "sinf32" => f.sin(),
                    "fabsf32" => f.abs(),
                    "cosf32" => f.cos(),
                    "sqrtf32" => f.sqrt(),
                    "expf32" => f.exp(),
                    "exp2f32" => f.exp2(),
                    "logf32" => f.ln(),
                    "log10f32" => f.log10(),
                    "log2f32" => f.log2(),
                    "floorf32" => f.floor(),
                    "ceilf32" => f.ceil(),
                    "truncf32" => f.trunc(),
                    _ => bug!(),
                };
                ecx.write_primval(dest, PrimVal::Bytes(f.to_bits() as u128), dest_layout.ty)?;
            }

            "sinf64" | "fabsf64" | "cosf64" | "sqrtf64" | "expf64" | "exp2f64" | "logf64" |
            "log10f64" | "log2f64" | "floorf64" | "ceilf64" | "truncf64" => {
                let f = ecx.value_to_primval(args[0])?.to_bytes()?;
                let f = f64::from_bits(f as u64);
                let f = match intrinsic_name {
                    "sinf64" => f.sin(),
                    "fabsf64" => f.abs(),
                    "cosf64" => f.cos(),
                    "sqrtf64" => f.sqrt(),
                    "expf64" => f.exp(),
                    "exp2f64" => f.exp2(),
                    "logf64" => f.ln(),
                    "log10f64" => f.log10(),
                    "log2f64" => f.log2(),
                    "floorf64" => f.floor(),
                    "ceilf64" => f.ceil(),
                    "truncf64" => f.trunc(),
                    _ => bug!(),
                };
                ecx.write_primval(dest, PrimVal::Bytes(f.to_bits() as u128), dest_layout.ty)?;
            }

            "assume" => {
                let cond = ecx.value_to_primval(args[0])?.to_bool()?;
                if !cond {
                    return err!(AssumptionNotHeld);
                }
            }

            "likely" | "unlikely" | "forget" => {}

            "align_offset" => {
                // FIXME: return a real value in case the target allocation has an
                // alignment bigger than the one requested
                ecx.write_primval(dest, PrimVal::Bytes(u128::max_value()), dest_layout.ty)?;
            },

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

            "transmute" => {
                let src_ty = substs.type_at(0);
                let _src_align = ecx.layout_of(src_ty)?.align;
                let ptr = ecx.force_allocation(dest)?.to_ptr()?;
                let dest_align = ecx.layout_of(substs.type_at(1))?.align;
                ecx.write_value_to_ptr(args[0].value, ptr.into(), dest_align, src_ty).unwrap();
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
        Ok(ecx
            .tcx
            .interpret_interner
            .cache_static(cid.instance.def_id()))
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


fn numeric_intrinsic<'tcx>(
    name: &str,
    bytes: u128,
    kind: PrimValKind,
) -> EvalResult<'tcx, PrimVal> {
    macro_rules! integer_intrinsic {
        ($method:ident) => ({
            use rustc::mir::interpret::PrimValKind::*;
            let result_bytes = match kind {
                I8 => (bytes as i8).$method() as u128,
                U8 => (bytes as u8).$method() as u128,
                I16 => (bytes as i16).$method() as u128,
                U16 => (bytes as u16).$method() as u128,
                I32 => (bytes as i32).$method() as u128,
                U32 => (bytes as u32).$method() as u128,
                I64 => (bytes as i64).$method() as u128,
                U64 => (bytes as u64).$method() as u128,
                I128 => (bytes as i128).$method() as u128,
                U128 => bytes.$method() as u128,
                _ => bug!("invalid `{}` argument: {:?}", name, bytes),
            };

            PrimVal::Bytes(result_bytes)
        });
    }

    let result_val = match name {
        "bswap" => integer_intrinsic!(swap_bytes),
        "ctlz" => integer_intrinsic!(leading_zeros),
        "ctpop" => integer_intrinsic!(count_ones),
        "cttz" => integer_intrinsic!(trailing_zeros),
        _ => bug!("not a numeric intrinsic: {}", name),
    };

    Ok(result_val)
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

pub fn const_variant_index<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    instance: ty::Instance<'tcx>,
    value: Value,
    ty: Ty<'tcx>,
) -> EvalResult<'tcx, usize> {
    trace!("const_variant_index: {:?}, {:?}, {:?}", instance, value, ty);
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
    ecx.read_discriminant_as_variant_index(place, ty)
}

pub fn const_eval_provider<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    key: ty::ParamEnvAnd<'tcx, GlobalId<'tcx>>,
) -> ::rustc::middle::const_val::EvalResult<'tcx> {
    trace!("const eval: {:?}", key);
    let cid = key.value;
    let def_id = cid.instance.def.def_id();

    if tcx.is_foreign_item(def_id) {
        let id = tcx.interpret_interner.cache_static(def_id);
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
