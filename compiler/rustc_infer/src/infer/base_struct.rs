use std::{iter, mem};
use crate::infer::sub::Sub;
use rustc_middle::ty;
use rustc_middle::ty::relate::{Cause, Relate, relate_generic_arg, RelateResult, TypeRelation};
use rustc_middle::ty::{Subst, SubstsRef, Ty, TyCtxt};
use rustc_span::def_id::DefId;

pub struct BaseStruct<'combine, 'infcx, 'tcx> {
    sub: Sub<'combine, 'infcx, 'tcx>,
}

impl<'combine, 'infcx, 'tcx> BaseStruct<'combine, 'infcx, 'tcx> {
    pub fn new(
        sub: Sub<'combine, 'infcx, 'tcx>,
    ) -> Self {
        BaseStruct { sub }
    }
}

impl<'tcx> TypeRelation<'tcx> for BaseStruct<'_, '_, 'tcx> {
    fn tag(&self) -> &'static str {
        "BaseStruct"
    }

    #[inline(always)]
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.sub.tcx()
    }

    #[inline(always)]
    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        self.sub.param_env()
    }

    #[inline(always)]
    fn a_is_expected(&self) -> bool {
        self.sub.a_is_expected()
    }

    #[inline(always)]
    fn with_cause<F, R>(&mut self, cause: Cause, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        let old_cause = mem::replace(&mut self.sub.fields.cause, Some(cause));
        let r = f(self);
        self.sub.fields.cause = old_cause;
        r
    }

    fn relate_item_substs(
        &mut self,
        item_def_id: DefId,
        a_subst: SubstsRef<'tcx>,
        b_subst: SubstsRef<'tcx>,
    ) -> RelateResult<'tcx, SubstsRef<'tcx>> {
        debug!(
            "relate_item_substs(item_def_id={:?}, a_subst={:?}, b_subst={:?})",
            item_def_id, a_subst, b_subst
        );

        let tcx = self.tcx();
        let variances = tcx.variances_of(item_def_id);

        let mut cached_ty = None;
        let params = iter::zip(a_subst, b_subst).enumerate().map(|(i, (a, b))| {
            let cached_ty =
                *cached_ty.get_or_insert_with(|| tcx.bound_type_of(item_def_id).subst(tcx, a_subst));
            relate_generic_arg(&mut self.sub, variances, cached_ty, a, b, i).or_else(|_| {
                Ok(b)
            })
        });

        tcx.mk_substs(params)
    }

    #[inline(always)]
    fn relate_with_variance<T: Relate<'tcx>>(
        &mut self,
        variance: ty::Variance,
        info: ty::VarianceDiagInfo<'tcx>,
        a: T,
        b: T,
    ) -> RelateResult<'tcx, T> {
        self.sub.relate_with_variance(variance, info, a, b)
    }

    #[inline(always)]
    #[instrument(skip(self), level = "debug")]
    fn tys(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        match (a.kind(), b.kind()) {
            (&ty::Adt(a_def, a_substs), &ty::Adt(b_def, b_substs)) if a_def == b_def => {
                let substs = self.relate_item_substs(a_def.did(), a_substs, b_substs)?;
                Ok(self.tcx().mk_adt(a_def, substs))
            }
            _ => bug!("not adt ty: {:?} and {:?}", a, b)
        }
    }

    #[inline(always)]
    fn regions(
        &mut self,
        a: ty::Region<'tcx>,
        b: ty::Region<'tcx>,
    ) -> RelateResult<'tcx, ty::Region<'tcx>> {
        self.sub.regions(a, b)
    }

    #[inline(always)]
    fn consts(
        &mut self,
        a: ty::Const<'tcx>,
        b: ty::Const<'tcx>,
    ) -> RelateResult<'tcx, ty::Const<'tcx>> {
        self.sub.consts(a, b)
    }

    #[inline(always)]
    fn binders<T>(
        &mut self,
        a: ty::Binder<'tcx, T>,
        b: ty::Binder<'tcx, T>,
    ) -> RelateResult<'tcx, ty::Binder<'tcx, T>>
    where
        T: Relate<'tcx>,
    {
        self.sub.binders(a, b)
    }
}
