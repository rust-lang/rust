use crate::ty::error::TypeError;
use crate::ty::relate::{self, Relate, RelateResult, TypeRelation};
use crate::ty::{self, InferConst, Ty, TyCtxt};

/// A type "A" *matches* "B" if the fresh types in B could be
/// substituted with values so as to make it equal to A. Matching is
/// intended to be used only on freshened types, and it basically
/// indicates if the non-freshened versions of A and B could have been
/// unified.
///
/// It is only an approximation. If it yields false, unification would
/// definitely fail, but a true result doesn't mean unification would
/// succeed. This is because we don't track the "side-constraints" on
/// type variables, nor do we track if the same freshened type appears
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

impl Match<'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, param_env: ty::ParamEnv<'tcx>) -> Match<'tcx> {
        Match { tcx, param_env }
    }
}

impl TypeRelation<'tcx> for Match<'tcx> {
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
        a: T,
        b: T,
    ) -> RelateResult<'tcx, T> {
        self.relate(a, b)
    }

    fn regions(
        &mut self,
        a: ty::Region<'tcx>,
        b: ty::Region<'tcx>,
    ) -> RelateResult<'tcx, ty::Region<'tcx>> {
        debug!("{}.regions({:?}, {:?})", self.tag(), a, b);
        Ok(a)
    }

    fn tys(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        debug!("{}.tys({:?}, {:?})", self.tag(), a, b);
        if a == b {
            return Ok(a);
        }

        match (a.kind(), b.kind()) {
            (
                _,
                &ty::Infer(ty::FreshTy(_))
                | &ty::Infer(ty::FreshIntTy(_))
                | &ty::Infer(ty::FreshFloatTy(_)),
            ) => Ok(a),

            (&ty::Infer(_), _) | (_, &ty::Infer(_)) => {
                Err(TypeError::Sorts(relate::expected_found(self, &a, &b)))
            }

            (&ty::Error(_), _) | (_, &ty::Error(_)) => Ok(self.tcx().ty_error()),

            _ => relate::super_relate_tys(self, a, b),
        }
    }

    fn consts(
        &mut self,
        a: &'tcx ty::Const<'tcx>,
        b: &'tcx ty::Const<'tcx>,
    ) -> RelateResult<'tcx, &'tcx ty::Const<'tcx>> {
        debug!("{}.consts({:?}, {:?})", self.tag(), a, b);
        if a == b {
            return Ok(a);
        }

        match (a.val, b.val) {
            (_, ty::ConstKind::Infer(InferConst::Fresh(_))) => {
                return Ok(a);
            }

            (ty::ConstKind::Infer(_), _) | (_, ty::ConstKind::Infer(_)) => {
                return Err(TypeError::ConstMismatch(relate::expected_found(self, &a, &b)));
            }

            _ => {}
        }

        relate::super_relate_consts(self, a, b)
    }

    fn binders<T>(
        &mut self,
        a: ty::Binder<T>,
        b: ty::Binder<T>,
    ) -> RelateResult<'tcx, ty::Binder<T>>
    where
        T: Relate<'tcx>,
    {
        Ok(ty::Binder::bind(self.relate(a.skip_binder(), b.skip_binder())?))
    }
}
