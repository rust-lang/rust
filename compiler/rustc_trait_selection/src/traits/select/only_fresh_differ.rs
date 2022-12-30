use rustc_middle::ty::error::TypeError;
use rustc_middle::ty::relate::{self, Relate, RelateResult, TypeRelation};
use rustc_middle::ty::{self, InferConst, Ty, TyCtxt};

/// Requires that the types can be unified by substituting fresh variables
/// for fresh variable only.
pub struct FreshDiffer<'tcx> {
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
}

impl<'tcx> FreshDiffer<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, param_env: ty::ParamEnv<'tcx>) -> Self {
        Self { tcx, param_env }
    }
}

impl<'tcx> TypeRelation<'tcx> for FreshDiffer<'tcx> {
    fn tag(&self) -> &'static str {
        "FreshDiffer"
    }
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn intercrate(&self) -> bool {
        false
    }

    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        self.param_env
    }
    fn a_is_expected(&self) -> bool {
        true
    } // irrelevant

    fn mark_ambiguous(&mut self) {
        bug!()
    }

    fn relate_with_variance<T: Relate<'tcx>>(
        &mut self,
        _: ty::Variance,
        _: ty::VarianceDiagInfo<'tcx>,
        a: T,
        b: T,
    ) -> RelateResult<'tcx, T> {
        self.relate(a, b)
    }

    #[instrument(skip(self), level = "debug")]
    fn regions(
        &mut self,
        a: ty::Region<'tcx>,
        b: ty::Region<'tcx>,
    ) -> RelateResult<'tcx, ty::Region<'tcx>> {
        Ok(a)
    }

    #[instrument(skip(self), level = "debug")]
    fn tys(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        if a == b {
            return Ok(a);
        }

        match (a.kind(), b.kind()) {
            (&ty::Infer(ty::FreshTy(_)), &ty::Infer(ty::FreshTy(_)))
            | (&ty::Infer(ty::FreshIntTy(_)), &ty::Infer(ty::FreshIntTy(_)))
            | (&ty::Infer(ty::FreshFloatTy(_)), &ty::Infer(ty::FreshFloatTy(_))) => Ok(a),

            (&ty::Infer(_), _) | (_, &ty::Infer(_)) => {
                Err(TypeError::Sorts(relate::expected_found(self, a, b)))
            }

            (&ty::Error(_), _) | (_, &ty::Error(_)) => Ok(self.tcx().ty_error()),

            _ => relate::super_relate_tys(self, a, b),
        }
    }

    fn consts(
        &mut self,
        a: ty::Const<'tcx>,
        b: ty::Const<'tcx>,
    ) -> RelateResult<'tcx, ty::Const<'tcx>> {
        debug!("{}.consts({:?}, {:?})", self.tag(), a, b);
        if a == b {
            return Ok(a);
        }

        match (a.kind(), b.kind()) {
            (_, ty::ConstKind::Infer(InferConst::Fresh(_))) => {
                return Ok(a);
            }

            (ty::ConstKind::Infer(_), _) | (_, ty::ConstKind::Infer(_)) => {
                return Err(TypeError::ConstMismatch(relate::expected_found(self, a, b)));
            }

            _ => {}
        }

        relate::super_relate_consts(self, a, b)
    }

    fn binders<T>(
        &mut self,
        a: ty::Binder<'tcx, T>,
        b: ty::Binder<'tcx, T>,
    ) -> RelateResult<'tcx, ty::Binder<'tcx, T>>
    where
        T: Relate<'tcx>,
    {
        Ok(a.rebind(self.relate(a.skip_binder(), b.skip_binder())?))
    }
}
