use rustc::traits::Reveal;
use rustc::ty::{self, TyCtxt, Ty, Instance, layout};
use rustc::mir;

use syntax::ast::Mutability;
use syntax::codemap::Span;

use super::{EvalResult, EvalError, EvalErrorKind, GlobalId, Lvalue, Value, PrimVal, EvalContext,
            StackPopCleanup, PtrAndAlign, MemoryKind, ValTy};

use rustc_const_math::ConstInt;

use std::fmt;
use std::error::Error;

pub fn eval_body_as_primval<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    instance: Instance<'tcx>,
) -> EvalResult<'tcx, (PrimVal, Ty<'tcx>)> {
    let limits = super::ResourceLimits::default();
    let mut ecx = EvalContext::<CompileTimeFunctionEvaluator>::new(tcx, limits, (), ());
    let cid = GlobalId {
        instance,
        promoted: None,
    };
    if ecx.tcx.has_attr(instance.def_id(), "linkage") {
        return Err(ConstEvalError::NotConst("extern global".to_string()).into());
    }

    let mir = ecx.load_mir(instance.def)?;
    if !ecx.globals.contains_key(&cid) {
        let size = ecx.type_size_with_substs(mir.return_ty, instance.substs)?
            .expect("unsized global");
        let align = ecx.type_align_with_substs(mir.return_ty, instance.substs)?;
        let ptr = ecx.memory.allocate(
            size,
            align,
            MemoryKind::UninitializedStatic,
        )?;
        let aligned = !ecx.is_packed(mir.return_ty)?;
        ecx.globals.insert(
            cid,
            PtrAndAlign {
                ptr: ptr.into(),
                aligned,
            },
        );
        let mutable = !mir.return_ty.is_freeze(
            ecx.tcx,
            ty::ParamEnv::empty(Reveal::All),
            mir.span,
        );
        let mutability = if mutable {
            Mutability::Mutable
        } else {
            Mutability::Immutable
        };
        let cleanup = StackPopCleanup::MarkStatic(mutability);
        let name = ty::tls::with(|tcx| tcx.item_path_str(instance.def_id()));
        trace!("const_eval: pushing stack frame for global: {}", name);
        ecx.push_stack_frame(
            instance,
            mir.span,
            mir,
            Lvalue::from_ptr(ptr),
            cleanup,
        )?;

        while ecx.step()? {}
    }
    let value = Value::ByRef(*ecx.globals.get(&cid).expect("global not cached"));
    let valty = ValTy {
        value,
        ty: mir.return_ty,
    };
    Ok((ecx.value_to_primval(valty)?, mir.return_ty))
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
        TyInt(IntTy::Is) => ConstInt::Isize(
            ConstIsize::new(prim as i128 as i64, tcx.sess.target.isize_ty)
                .expect("miri should already have errored"),
        ),
        TyUint(UintTy::U8) => ConstInt::U8(prim as u8),
        TyUint(UintTy::U16) => ConstInt::U16(prim as u16),
        TyUint(UintTy::U32) => ConstInt::U32(prim as u32),
        TyUint(UintTy::U64) => ConstInt::U64(prim as u64),
        TyUint(UintTy::U128) => ConstInt::U128(prim),
        TyUint(UintTy::Us) => ConstInt::Usize(
            ConstUsize::new(prim as u64, tcx.sess.target.usize_ty)
                .expect("miri should already have errored"),
        ),
        _ => {
            return Err(
                ConstEvalError::NeedsRfc(
                    "evaluating anything other than isize/usize during typeck".to_string(),
                ).into(),
            )
        }
    })
}

struct CompileTimeFunctionEvaluator;

impl<'tcx> Into<EvalError<'tcx>> for ConstEvalError {
    fn into(self) -> EvalError<'tcx> {
        EvalErrorKind::MachineError(Box::new(self)).into()
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

    fn cause(&self) -> Option<&Error> {
        None
    }
}

impl<'tcx> super::Machine<'tcx> for CompileTimeFunctionEvaluator {
    type Data = ();
    type MemoryData = ();
    type MemoryKinds = !;
    fn eval_fn_call<'a>(
        ecx: &mut EvalContext<'a, 'tcx, Self>,
        instance: ty::Instance<'tcx>,
        destination: Option<(Lvalue, mir::BasicBlock)>,
        _args: &[ValTy<'tcx>],
        span: Span,
        _sig: ty::FnSig<'tcx>,
    ) -> EvalResult<'tcx, bool> {
        if !ecx.tcx.is_const_fn(instance.def_id()) {
            return Err(
                ConstEvalError::NotConst(format!("calling non-const fn `{}`", instance)).into(),
            );
        }
        let mir = match ecx.load_mir(instance.def) {
            Ok(mir) => mir,
            Err(EvalError { kind: EvalErrorKind::NoMirFor(path), .. }) => {
                // some simple things like `malloc` might get accepted in the future
                return Err(
                    ConstEvalError::NeedsRfc(format!("calling extern function `{}`", path))
                        .into(),
                );
            }
            Err(other) => return Err(other),
        };
        let (return_lvalue, return_to_block) = match destination {
            Some((lvalue, block)) => (lvalue, StackPopCleanup::Goto(block)),
            None => (Lvalue::undef(), StackPopCleanup::None),
        };

        ecx.push_stack_frame(
            instance,
            span,
            mir,
            return_lvalue,
            return_to_block,
        )?;

        Ok(false)
    }

    fn call_intrinsic<'a>(
        _ecx: &mut EvalContext<'a, 'tcx, Self>,
        _instance: ty::Instance<'tcx>,
        _args: &[ValTy<'tcx>],
        _dest: Lvalue,
        _dest_ty: Ty<'tcx>,
        _dest_layout: &'tcx layout::Layout,
        _target: mir::BasicBlock,
    ) -> EvalResult<'tcx> {
        Err(
            ConstEvalError::NeedsRfc("calling intrinsics".to_string()).into(),
        )
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
        _ty: ty::Ty<'tcx>,
        _dest: Lvalue,
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
