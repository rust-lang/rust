use rustc_infer::infer::relate::{
    self, Relate, RelateResult, TypeRelation, structurally_relate_tys,
};
use rustc_middle::ty::error::{ExpectedFound, TypeError};
use rustc_middle::ty::{self, InferConst, Ty, TyCtxt};
use tracing::instrument;

/// A type "A" *matches* "B" if the fresh types in B could be
/// instantiated with values so as to make it equal to A. Matching is
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
pub(crate) struct MatchAgainstFreshVars<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> MatchAgainstFreshVars<'tcx> {
    pub(crate) fn new(tcx: TyCtxt<'tcx>) -> MatchAgainstFreshVars<'tcx> {
        MatchAgainstFreshVars { tcx }
    }
}

impl<'tcx> TypeRelation<TyCtxt<'tcx>> for MatchAgainstFreshVars<'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn relate_with_variance<T: Relate<TyCtxt<'tcx>>>(
        &mut self,
        _: ty::Variance,
        _: ty::VarianceDiagInfo<TyCtxt<'tcx>>,
        a: T,
        b: T,
    ) -> RelateResult<'tcx, T> {
        self.relate(a, b)
    }

    #[instrument(skip(self), level = "trace")]
    fn regions(
        &mut self,
        a: ty::Region<'tcx>,
        _b: ty::Region<'tcx>,
    ) -> RelateResult<'tcx, ty::Region<'tcx>> {
        Ok(a)
    }

    #[instrument(skip(self), level = "trace")]
    fn tys(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
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
                Err(TypeError::Sorts(ExpectedFound::new(a, b)))
            }

            (&ty::Error(guar), _) | (_, &ty::Error(guar)) => Ok(Ty::new_error(self.cx(), guar)),

            _ => structurally_relate_tys(self, a, b),
        }
    }

    #[instrument(skip(self), level = "trace")]
    fn consts(
        &mut self,
        a: ty::Const<'tcx>,
        b: ty::Const<'tcx>,
    ) -> RelateResult<'tcx, ty::Const<'tcx>> {
        if a == b {
            return Ok(a);
        }

        match (a.kind(), b.kind()) {
            (_, ty::ConstKind::Infer(InferConst::Fresh(_))) => {
                return Ok(a);
            }

            (ty::ConstKind::Infer(_), _) | (_, ty::ConstKind::Infer(_)) => {
                return Err(TypeError::ConstMismatch(ExpectedFound::new(a, b)));
            }

            _ => {}
        }

        relate::structurally_relate_consts(self, a, b)
    }

    fn binders<T>(
        &mut self,
        a: ty::Binder<'tcx, T>,
        b: ty::Binder<'tcx, T>,
    ) -> RelateResult<'tcx, ty::Binder<'tcx, T>>
    where
        T: Relate<TyCtxt<'tcx>>,
    {
        Ok(a.rebind(self.relate(a.skip_binder(), b.skip_binder())?))
    }
}
