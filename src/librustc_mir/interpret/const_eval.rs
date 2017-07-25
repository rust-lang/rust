use rustc::traits::Reveal;
use rustc::ty::{self, TyCtxt, Ty, Instance};
use rustc::mir;

use syntax::ast::Mutability;

use super::{
    EvalResult, EvalError,
    Global, GlobalId, Lvalue,
    PrimVal,
    EvalContext, StackPopCleanup,
};

use rustc_const_math::ConstInt;

use std::fmt;
use std::error::Error;

pub fn eval_body_as_primval<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    instance: Instance<'tcx>,
) -> EvalResult<'tcx, (PrimVal, Ty<'tcx>)> {
    let limits = super::ResourceLimits::default();
    let mut ecx = EvalContext::<Evaluator>::new(tcx, limits, (), ());
    let cid = GlobalId { instance, promoted: None };
    if ecx.tcx.has_attr(instance.def_id(), "linkage") {
        return Err(ConstEvalError::NotConst("extern global".to_string()).into());
    }
    
    let mir = ecx.load_mir(instance.def)?;
    if !ecx.globals.contains_key(&cid) {
        ecx.globals.insert(cid, Global::uninitialized(mir.return_ty));
        let mutable = !mir.return_ty.is_freeze(
                ecx.tcx,
                ty::ParamEnv::empty(Reveal::All),
                mir.span);
        let mutability = if mutable {
            Mutability::Mutable
        } else {
            Mutability::Immutable
        };
        let cleanup = StackPopCleanup::MarkStatic(mutability);
        let name = ty::tls::with(|tcx| tcx.item_path_str(instance.def_id()));
        trace!("pushing stack frame for global: {}", name);
        ecx.push_stack_frame(
            instance,
            mir.span,
            mir,
            Lvalue::Global(cid),
            cleanup,
        )?;

        while ecx.step()? {}
    }
    let value = ecx.globals.get(&cid).expect("global not cached").value;
    Ok((ecx.value_to_primval(value, mir.return_ty)?, mir.return_ty))
}

pub fn eval_body_as_integer<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    instance: Instance<'tcx>,
) -> EvalResult<'tcx, ConstInt> {
    let (prim, ty) = eval_body_as_primval(tcx, instance)?;
    let prim = prim.to_bytes()?;
    use syntax::ast::{IntTy, UintTy};
    use rustc::ty::TypeVariants::*;
    use rustc_const_math::{ConstIsize, ConstUsize};
    Ok(match ty.sty {
        TyInt(IntTy::I8) => ConstInt::I8(prim as i128 as i8),
        TyInt(IntTy::I16) => ConstInt::I16(prim as i128 as i16),
        TyInt(IntTy::I32) => ConstInt::I32(prim as i128 as i32),
        TyInt(IntTy::I64) => ConstInt::I64(prim as i128 as i64),
        TyInt(IntTy::I128) => ConstInt::I128(prim as i128),
        TyInt(IntTy::Is) => ConstInt::Isize(ConstIsize::new(prim as i128 as i64, tcx.sess.target.int_type).expect("miri should already have errored")),
        TyUint(UintTy::U8) => ConstInt::U8(prim as u8),
        TyUint(UintTy::U16) => ConstInt::U16(prim as u16),
        TyUint(UintTy::U32) => ConstInt::U32(prim as u32),
        TyUint(UintTy::U64) => ConstInt::U64(prim as u64),
        TyUint(UintTy::U128) => ConstInt::U128(prim),
        TyUint(UintTy::Us) => ConstInt::Usize(ConstUsize::new(prim as u64, tcx.sess.target.uint_type).expect("miri should already have errored")),
        _ => return Err(ConstEvalError::NeedsRfc("evaluating anything other than isize/usize during typeck".to_string()).into()),
    })
}

struct Evaluator;

impl<'tcx> Into<EvalError<'tcx>> for ConstEvalError {
    fn into(self) -> EvalError<'tcx> {
        EvalError::MachineError(Box::new(self))
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
            NeedsRfc(ref msg) =>
                write!(f, "\"{}\" needs an rfc before being allowed inside constants", msg),
            NotConst(ref msg) =>
                write!(f, "Cannot evaluate within constants: \"{}\"", msg),
        }
    }
}

impl Error for ConstEvalError {
    fn description(&self) -> &str {
        use self::ConstEvalError::*;
        match *self {
            NeedsRfc(_) =>
                "this feature needs an rfc before being allowed inside constants",
            NotConst(_) =>
                "this feature is not compatible with constant evaluation",
        }
    }

    fn cause(&self) -> Option<&Error> {
        None
    }
}

impl<'tcx> super::Machine<'tcx> for Evaluator {
    type Data = ();
    type MemoryData = ();
    fn call_missing_fn<'a>(
        _ecx: &mut EvalContext<'a, 'tcx, Self>,
        _instance: ty::Instance<'tcx>,
        _destination: Option<(Lvalue<'tcx>, mir::BasicBlock)>,
        _arg_operands: &[mir::Operand<'tcx>],
        _sig: ty::FnSig<'tcx>,
        path: String,
    ) -> EvalResult<'tcx> {
        // some simple things like `malloc` might get accepted in the future
        Err(ConstEvalError::NeedsRfc(format!("calling extern function `{}`", path)).into())
    }

    fn ptr_op<'a>(
        _ecx: &EvalContext<'a, 'tcx, Self>,
        _bin_op: mir::BinOp,
        _left: PrimVal,
        _left_ty: Ty<'tcx>,
        _right: PrimVal,
        _right_ty: Ty<'tcx>,
    ) -> EvalResult<'tcx, Option<(PrimVal, bool)>> {
        Err(ConstEvalError::NeedsRfc("Pointer arithmetic or comparison".to_string()).into())
    }

    fn check_non_const_fn_call(instance: ty::Instance<'tcx>) -> EvalResult<'tcx> {
        return Err(ConstEvalError::NotConst(format!("calling non-const fn `{}`", instance)).into());
    }
}
