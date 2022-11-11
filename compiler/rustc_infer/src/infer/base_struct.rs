use crate::infer::at::{At, ToTrace};
use crate::infer::sub::Sub;
use crate::infer::{InferOk, InferResult};
use rustc_middle::ty;
use rustc_middle::ty::relate::{relate_generic_arg, RelateResult, TypeRelation};
use rustc_middle::ty::{GenericArg, SubstsRef, Ty};
use std::iter;

pub fn base_struct<'a, 'tcx>(at: At<'a, 'tcx>, a: Ty<'tcx>, b: Ty<'tcx>) -> InferResult<'tcx, ()> {
    let trace = ToTrace::to_trace(at.infcx.tcx, at.cause, true, a, b);
    at.infcx.commit_if_ok(|_| {
        let mut fields = at.infcx.combine_fields(trace, at.param_env, at.define_opaque_types);
        let mut sub = Sub::new(&mut fields, true);
        base_struct_tys(&mut sub, a, b)
            .map(move |_| InferOk { value: (), obligations: fields.obligations })
    })
}

pub fn base_struct_tys<'tcx>(
    sub: &mut Sub<'_, '_, 'tcx>,
    a: Ty<'tcx>,
    b: Ty<'tcx>,
) -> RelateResult<'tcx, ()> {
    match (a.kind(), b.kind()) {
        (&ty::Adt(a_def, a_substs), &ty::Adt(b_def, b_substs)) if a_def == b_def => {
            base_struct_substs(sub, a, a_substs, b_substs)?;
            Ok(())
        }
        _ => bug!("not adt ty: {:?} and {:?}", a, b),
    }
}

fn base_struct_substs<'tcx>(
    sub: &mut Sub<'_, '_, 'tcx>,
    adt_ty: Ty<'tcx>,
    a_subst: SubstsRef<'tcx>,
    b_subst: SubstsRef<'tcx>,
) -> RelateResult<'tcx, ()> {
    let tcx = sub.tcx();
    let variances = tcx.variances_of(adt_ty.ty_adt_def().expect("not a adt ty!").did());

    iter::zip(a_subst, b_subst).enumerate().for_each(|(i, (a, b))| {
        let _arg: RelateResult<'tcx, GenericArg<'tcx>> =
            relate_generic_arg(sub, variances, adt_ty, a, b, i).or_else(|_| Ok(b));
    });

    Ok(())
}
