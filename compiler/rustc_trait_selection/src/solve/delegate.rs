use std::ops::Deref;

use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def_id::DefId;
use rustc_infer::infer::canonical::query_response::make_query_region_constraints;
use rustc_infer::infer::canonical::{
    Canonical, CanonicalExt as _, CanonicalQueryInput, CanonicalVarInfo, CanonicalVarValues,
};
use rustc_infer::infer::{InferCtxt, RegionVariableOrigin, TyCtxtInferExt};
use rustc_infer::traits::solve::Goal;
use rustc_infer::traits::{ObligationCause, Reveal};
use rustc_middle::ty::fold::TypeFoldable;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeVisitableExt as _};
use rustc_span::{DUMMY_SP, ErrorGuaranteed, Span};
use rustc_type_ir::solve::{Certainty, NoSolution, SolverMode};
use tracing::trace;

use crate::traits::specialization_graph;

#[repr(transparent)]
pub struct SolverDelegate<'tcx>(InferCtxt<'tcx>);

impl<'a, 'tcx> From<&'a InferCtxt<'tcx>> for &'a SolverDelegate<'tcx> {
    fn from(infcx: &'a InferCtxt<'tcx>) -> Self {
        // SAFETY: `repr(transparent)`
        unsafe { std::mem::transmute(infcx) }
    }
}

impl<'tcx> Deref for SolverDelegate<'tcx> {
    type Target = InferCtxt<'tcx>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'tcx> rustc_next_trait_solver::delegate::SolverDelegate for SolverDelegate<'tcx> {
    type Infcx = InferCtxt<'tcx>;
    type Interner = TyCtxt<'tcx>;

    fn cx(&self) -> TyCtxt<'tcx> {
        self.0.tcx
    }

    type Span = Span;

    fn build_with_canonical<V>(
        interner: TyCtxt<'tcx>,
        solver_mode: SolverMode,
        canonical: &CanonicalQueryInput<'tcx, V>,
    ) -> (Self, V, CanonicalVarValues<'tcx>)
    where
        V: TypeFoldable<TyCtxt<'tcx>>,
    {
        let (infcx, value, vars) = interner
            .infer_ctxt()
            .with_next_trait_solver(true)
            .intercrate(match solver_mode {
                SolverMode::Normal => false,
                SolverMode::Coherence => true,
            })
            .build_with_canonical(DUMMY_SP, canonical);
        (SolverDelegate(infcx), value, vars)
    }

    fn fresh_var_for_kind_with_span(
        &self,
        arg: ty::GenericArg<'tcx>,
        span: Span,
    ) -> ty::GenericArg<'tcx> {
        match arg.unpack() {
            ty::GenericArgKind::Lifetime(_) => {
                self.next_region_var(RegionVariableOrigin::MiscVariable(span)).into()
            }
            ty::GenericArgKind::Type(_) => self.next_ty_var(span).into(),
            ty::GenericArgKind::Const(_) => self.next_const_var(span).into(),
        }
    }

    fn leak_check(&self, max_input_universe: ty::UniverseIndex) -> Result<(), NoSolution> {
        self.0.leak_check(max_input_universe, None).map_err(|_| NoSolution)
    }

    fn try_const_eval_resolve(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        unevaluated: ty::UnevaluatedConst<'tcx>,
    ) -> Option<ty::Const<'tcx>> {
        use rustc_middle::mir::interpret::ErrorHandled;
        match self.const_eval_resolve(param_env, unevaluated, DUMMY_SP) {
            Ok(Ok(val)) => Some(ty::Const::new_value(
                self.tcx,
                val,
                self.tcx.type_of(unevaluated.def).instantiate(self.tcx, unevaluated.args),
            )),
            Ok(Err(_)) | Err(ErrorHandled::TooGeneric(_)) => None,
            Err(ErrorHandled::Reported(e, _)) => Some(ty::Const::new_error(self.tcx, e.into())),
        }
    }

    fn well_formed_goals(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        arg: ty::GenericArg<'tcx>,
    ) -> Option<Vec<Goal<'tcx, ty::Predicate<'tcx>>>> {
        crate::traits::wf::unnormalized_obligations(&self.0, param_env, arg).map(|obligations| {
            obligations.into_iter().map(|obligation| obligation.into()).collect()
        })
    }

    fn clone_opaque_types_for_query_response(&self) -> Vec<(ty::OpaqueTypeKey<'tcx>, Ty<'tcx>)> {
        self.0.clone_opaque_types_for_query_response()
    }

    fn make_deduplicated_outlives_constraints(
        &self,
    ) -> Vec<ty::OutlivesPredicate<'tcx, ty::GenericArg<'tcx>>> {
        // Cannot use `take_registered_region_obligations` as we may compute the response
        // inside of a `probe` whenever we have multiple choices inside of the solver.
        let region_obligations = self.0.inner.borrow().region_obligations().to_owned();
        let region_constraints = self.0.with_region_constraints(|region_constraints| {
            make_query_region_constraints(
                self.tcx,
                region_obligations
                    .iter()
                    .map(|r_o| (r_o.sup_type, r_o.sub_region, r_o.origin.to_constraint_category())),
                region_constraints,
            )
        });

        assert_eq!(region_constraints.member_constraints, vec![]);

        let mut seen = FxHashSet::default();
        region_constraints
            .outlives
            .into_iter()
            .filter(|&(outlives, _)| seen.insert(outlives))
            .map(|(outlives, _)| outlives)
            .collect()
    }

    fn instantiate_canonical<V>(
        &self,
        canonical: Canonical<'tcx, V>,
        values: CanonicalVarValues<'tcx>,
    ) -> V
    where
        V: TypeFoldable<TyCtxt<'tcx>>,
    {
        canonical.instantiate(self.tcx, &values)
    }

    fn instantiate_canonical_var_with_infer(
        &self,
        cv_info: CanonicalVarInfo<'tcx>,
        universe_map: impl Fn(ty::UniverseIndex) -> ty::UniverseIndex,
    ) -> ty::GenericArg<'tcx> {
        self.0.instantiate_canonical_var(DUMMY_SP, cv_info, universe_map)
    }

    fn insert_hidden_type(
        &self,
        opaque_type_key: ty::OpaqueTypeKey<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        hidden_ty: Ty<'tcx>,
        goals: &mut Vec<Goal<'tcx, ty::Predicate<'tcx>>>,
    ) -> Result<(), NoSolution> {
        self.0
            .insert_hidden_type(opaque_type_key, DUMMY_SP, param_env, hidden_ty, goals)
            .map_err(|_| NoSolution)
    }

    fn add_item_bounds_for_hidden_type(
        &self,
        def_id: DefId,
        args: ty::GenericArgsRef<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        hidden_ty: Ty<'tcx>,
        goals: &mut Vec<Goal<'tcx, ty::Predicate<'tcx>>>,
    ) {
        self.0.add_item_bounds_for_hidden_type(def_id, args, param_env, hidden_ty, goals);
    }

    fn inject_new_hidden_type_unchecked(&self, key: ty::OpaqueTypeKey<'tcx>, hidden_ty: Ty<'tcx>) {
        self.0.inject_new_hidden_type_unchecked(key, ty::OpaqueHiddenType {
            ty: hidden_ty,
            span: DUMMY_SP,
        })
    }

    fn reset_opaque_types(&self) {
        let _ = self.take_opaque_types();
    }

    fn fetch_eligible_assoc_item(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        goal_trait_ref: ty::TraitRef<'tcx>,
        trait_assoc_def_id: DefId,
        impl_def_id: DefId,
    ) -> Result<Option<DefId>, NoSolution> {
        let node_item = specialization_graph::assoc_def(self.tcx, impl_def_id, trait_assoc_def_id)
            .map_err(|ErrorGuaranteed { .. }| NoSolution)?;

        let eligible = if node_item.is_final() {
            // Non-specializable items are always projectable.
            true
        } else {
            // Only reveal a specializable default if we're past type-checking
            // and the obligation is monomorphic, otherwise passes such as
            // transmute checking and polymorphic MIR optimizations could
            // get a result which isn't correct for all monomorphizations.
            if param_env.reveal() == Reveal::All {
                let poly_trait_ref = self.resolve_vars_if_possible(goal_trait_ref);
                !poly_trait_ref.still_further_specializable()
            } else {
                trace!(?node_item.item.def_id, "not eligible due to default");
                false
            }
        };

        // FIXME: Check for defaultness here may cause diagnostics problems.
        if eligible { Ok(Some(node_item.item.def_id)) } else { Ok(None) }
    }

    // FIXME: This actually should destructure the `Result` we get from transmutability and
    // register candidates. We probably need to register >1 since we may have an OR of ANDs.
    fn is_transmutable(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        dst: Ty<'tcx>,
        src: Ty<'tcx>,
        assume: ty::Const<'tcx>,
    ) -> Result<Certainty, NoSolution> {
        // Erase regions because we compute layouts in `rustc_transmute`,
        // which will ICE for region vars.
        let (dst, src) = self.tcx.erase_regions((dst, src));

        let Some(assume) = rustc_transmute::Assume::from_const(self.tcx, param_env, assume) else {
            return Err(NoSolution);
        };

        // FIXME(transmutability): This really should be returning nested goals for `Answer::If*`
        match rustc_transmute::TransmuteTypeEnv::new(&self.0).is_transmutable(
            ObligationCause::dummy(),
            rustc_transmute::Types { src, dst },
            assume,
        ) {
            rustc_transmute::Answer::Yes => Ok(Certainty::Yes),
            rustc_transmute::Answer::No(_) | rustc_transmute::Answer::If(_) => Err(NoSolution),
        }
    }
}
