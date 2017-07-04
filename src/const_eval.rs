use rustc::hir::def_id::DefId;
use rustc::traits::Reveal;
use rustc::ty::subst::Substs;
use rustc::ty::{self, TyCtxt};

use error::{EvalError, EvalResult};
use lvalue::{Global, GlobalId, Lvalue};
use rustc_const_math::ConstInt;
use eval_context::{EvalContext, StackPopCleanup};

pub fn eval_body_as_integer<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    (def_id, substs): (DefId, &'tcx Substs<'tcx>),
) -> EvalResult<'tcx, ConstInt> {
    let limits = ::ResourceLimits::default();
    let mut ecx = EvalContext::new(tcx, limits);
    let instance = ecx.resolve_associated_const(def_id, substs);
    let cid = GlobalId { instance, promoted: None };
    if ecx.tcx.has_attr(def_id, "linkage") {
        return Err(EvalError::NotConst("extern global".to_string()));
    }
    
    let mir = ecx.load_mir(instance.def)?;
    if !ecx.globals.contains_key(&cid) {
        ecx.globals.insert(cid, Global::uninitialized(mir.return_ty));
        let mutable = !mir.return_ty.is_freeze(
                ecx.tcx,
                ty::ParamEnv::empty(Reveal::All),
                mir.span);
        let cleanup = StackPopCleanup::MarkStatic(mutable);
        let name = ty::tls::with(|tcx| tcx.item_path_str(def_id));
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
    let prim = ecx.value_to_primval(value, mir.return_ty)?.to_bytes()?;
    use syntax::ast::{IntTy, UintTy};
    use rustc::ty::TypeVariants::*;
    use rustc_const_math::{ConstIsize, ConstUsize};
    Ok(match mir.return_ty.sty {
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
        _ => return Err(EvalError::NeedsRfc("evaluating anything other than isize/usize during typeck".to_string())),
    })
}
