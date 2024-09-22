///! Definition of `InferCtxtLike` from the librarified type layer.
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_middle::traits::ObligationCause;
use rustc_middle::traits::solve::{Goal, NoSolution, SolverMode};
use rustc_middle::ty::fold::TypeFoldable;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::DUMMY_SP;
use rustc_type_ir::InferCtxtLike;
use rustc_type_ir::relate::Relate;

use super::{BoundRegionConversionTime, InferCtxt, SubregionOrigin};

impl<'tcx> InferCtxtLike for InferCtxt<'tcx> {
    type Interner = TyCtxt<'tcx>;

    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn solver_mode(&self) -> ty::solve::SolverMode {
        match self.intercrate {
            true => SolverMode::Coherence,
            false => SolverMode::Normal,
        }
    }

    fn universe(&self) -> ty::UniverseIndex {
        self.universe()
    }

    fn create_next_universe(&self) -> ty::UniverseIndex {
        self.create_next_universe()
    }

    fn universe_of_ty(&self, vid: ty::TyVid) -> Option<ty::UniverseIndex> {
        match self.probe_ty_var(vid) {
            Err(universe) => Some(universe),
            Ok(_) => None,
        }
    }

    fn universe_of_lt(&self, lt: ty::RegionVid) -> Option<ty::UniverseIndex> {
        match self.inner.borrow_mut().unwrap_region_constraints().probe_value(lt) {
            Err(universe) => Some(universe),
            Ok(_) => None,
        }
    }

    fn universe_of_ct(&self, ct: ty::ConstVid) -> Option<ty::UniverseIndex> {
        match self.probe_const_var(ct) {
            Err(universe) => Some(universe),
            Ok(_) => None,
        }
    }

    fn root_ty_var(&self, var: ty::TyVid) -> ty::TyVid {
        self.root_var(var)
    }

    fn root_const_var(&self, var: ty::ConstVid) -> ty::ConstVid {
        self.root_const_var(var)
    }

    fn opportunistic_resolve_ty_var(&self, vid: ty::TyVid) -> Ty<'tcx> {
        match self.probe_ty_var(vid) {
            Ok(ty) => ty,
            Err(_) => Ty::new_var(self.tcx, self.root_var(vid)),
        }
    }

    fn opportunistic_resolve_int_var(&self, vid: ty::IntVid) -> Ty<'tcx> {
        self.opportunistic_resolve_int_var(vid)
    }

    fn opportunistic_resolve_float_var(&self, vid: ty::FloatVid) -> Ty<'tcx> {
        self.opportunistic_resolve_float_var(vid)
    }

    fn opportunistic_resolve_ct_var(&self, vid: ty::ConstVid) -> ty::Const<'tcx> {
        match self.probe_const_var(vid) {
            Ok(ct) => ct,
            Err(_) => ty::Const::new_var(self.tcx, self.root_const_var(vid)),
        }
    }

    fn opportunistic_resolve_effect_var(&self, vid: ty::EffectVid) -> ty::Const<'tcx> {
        match self.probe_effect_var(vid) {
            Some(ct) => ct,
            None => {
                ty::Const::new_infer(self.tcx, ty::InferConst::EffectVar(self.root_effect_var(vid)))
            }
        }
    }

    fn opportunistic_resolve_lt_var(&self, vid: ty::RegionVid) -> ty::Region<'tcx> {
        self.inner.borrow_mut().unwrap_region_constraints().opportunistic_resolve_var(self.tcx, vid)
    }

    fn defining_opaque_types(&self) -> &'tcx ty::List<LocalDefId> {
        self.defining_opaque_types()
    }

    fn next_ty_infer(&self) -> Ty<'tcx> {
        self.next_ty_var(DUMMY_SP)
    }

    fn next_const_infer(&self) -> ty::Const<'tcx> {
        self.next_const_var(DUMMY_SP)
    }

    fn fresh_args_for_item(&self, def_id: DefId) -> ty::GenericArgsRef<'tcx> {
        self.fresh_args_for_item(DUMMY_SP, def_id)
    }

    fn instantiate_binder_with_infer<T: TypeFoldable<TyCtxt<'tcx>> + Copy>(
        &self,
        value: ty::Binder<'tcx, T>,
    ) -> T {
        self.instantiate_binder_with_fresh_vars(
            DUMMY_SP,
            BoundRegionConversionTime::HigherRankedType,
            value,
        )
    }

    fn enter_forall<T: TypeFoldable<TyCtxt<'tcx>> + Copy, U>(
        &self,
        value: ty::Binder<'tcx, T>,
        f: impl FnOnce(T) -> U,
    ) -> U {
        self.enter_forall(value, f)
    }

    fn relate<T: Relate<TyCtxt<'tcx>>>(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        lhs: T,
        variance: ty::Variance,
        rhs: T,
    ) -> Result<Vec<Goal<'tcx, ty::Predicate<'tcx>>>, NoSolution> {
        self.at(&ObligationCause::dummy(), param_env).relate_no_trace(lhs, variance, rhs)
    }

    fn eq_structurally_relating_aliases<T: Relate<TyCtxt<'tcx>>>(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        lhs: T,
        rhs: T,
    ) -> Result<Vec<Goal<'tcx, ty::Predicate<'tcx>>>, NoSolution> {
        self.at(&ObligationCause::dummy(), param_env)
            .eq_structurally_relating_aliases_no_trace(lhs, rhs)
    }

    fn shallow_resolve(&self, ty: Ty<'tcx>) -> Ty<'tcx> {
        self.shallow_resolve(ty)
    }

    fn resolve_vars_if_possible<T>(&self, value: T) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        self.resolve_vars_if_possible(value)
    }

    fn probe<T>(&self, probe: impl FnOnce() -> T) -> T {
        self.probe(|_| probe())
    }

    fn sub_regions(&self, sub: ty::Region<'tcx>, sup: ty::Region<'tcx>) {
        self.sub_regions(SubregionOrigin::RelateRegionParamBound(DUMMY_SP, None), sub, sup)
    }

    fn register_ty_outlives(&self, ty: Ty<'tcx>, r: ty::Region<'tcx>) {
        self.register_region_obligation_with_cause(ty, r, &ObligationCause::dummy());
    }
}
