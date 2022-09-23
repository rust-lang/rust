use crate::ty::error::TypeError;
use crate::ty::relate::{self, Relate, RelateResult, TypeRelation};
use crate::ty::{self, Ty, TyCtxt};

/// A type "A" *matches* "B" if the inference variables in B could be
/// substituted with values so as to make it equal to A.
/// Matching  basically indicates whether A would contain B.
///
/// It is only an approximation. If it yields false, unification would
/// definitely fail, but a true result doesn't mean unification would
/// succeed. This is because we don't track the "side-constraints" on
/// type variables, nor do we track if the same inference variable appears
/// more than once. To some extent these approximations could be
/// fixed, given effort.
///
/// Like subtyping, matching is really a binary relation, so the only
/// important thing about the result is Ok/Err. Also, matching never
/// affects any type variables or unification state.
pub struct Match<'tcx> {
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
}

impl<'tcx> Match<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, param_env: ty::ParamEnv<'tcx>) -> Match<'tcx> {
        Match { tcx, param_env }
    }
}

impl<'tcx> TypeRelation<'tcx> for Match<'tcx> {
    fn tag(&self) -> &'static str {
        "Match"
    }
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }
    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        self.param_env
    }
    fn a_is_expected(&self) -> bool {
        true
    } // irrelevant

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
            (_, &ty::Infer(_)) => Ok(a),

            (&ty::Infer(_), _) => Err(TypeError::Sorts(relate::expected_found(self, a, b))),

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
            (_, ty::ConstKind::Infer(_)) => {
                return Ok(a);
            }

            (ty::ConstKind::Infer(_), _) => {
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
