use std::ops::Deref;

use rustc_data_structures::fx::FxHashSet;
use rustc_hir::LangItem;
use rustc_hir::def_id::{CRATE_DEF_ID, DefId};
use rustc_infer::infer::canonical::query_response::make_query_region_constraints;
use rustc_infer::infer::canonical::{
    Canonical, CanonicalExt as _, CanonicalQueryInput, CanonicalVarKind, CanonicalVarValues,
};
use rustc_infer::infer::{InferCtxt, RegionVariableOrigin, SubregionOrigin, TyCtxtInferExt};
use rustc_infer::traits::solve::Goal;
use rustc_middle::traits::query::NoSolution;
use rustc_middle::traits::solve::Certainty;
use rustc_middle::ty::{self, Ty, TyCtxt, TypeFoldable, TypeVisitableExt as _, TypingMode};
use rustc_next_trait_solver::solve::HasChanged;
use rustc_span::{DUMMY_SP, ErrorGuaranteed, Span};

use crate::traits::{EvaluateConstErr, ObligationCause, specialization_graph};

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

    fn build_with_canonical<V>(
        interner: TyCtxt<'tcx>,
        canonical: &CanonicalQueryInput<'tcx, V>,
    ) -> (Self, V, CanonicalVarValues<'tcx>)
    where
        V: TypeFoldable<TyCtxt<'tcx>>,
    {
        let (infcx, value, vars) = interner
            .infer_ctxt()
            .with_next_trait_solver(true)
            .build_with_canonical(DUMMY_SP, canonical);
        (SolverDelegate(infcx), value, vars)
    }

    fn compute_goal_fast_path(
        &self,
        goal: Goal<'tcx, ty::Predicate<'tcx>>,
        span: Span,
    ) -> Option<HasChanged> {
        let pred = goal.predicate.kind();
        match pred.no_bound_vars()? {
            ty::PredicateKind::DynCompatible(def_id) if self.0.tcx.is_dyn_compatible(def_id) => {
                Some(HasChanged::No)
            }
            ty::PredicateKind::Clause(ty::ClauseKind::RegionOutlives(outlives)) => {
                self.0.sub_regions(
                    SubregionOrigin::RelateRegionParamBound(span, None),
                    outlives.1,
                    outlives.0,
                );
                Some(HasChanged::No)
            }
            ty::PredicateKind::Clause(ty::ClauseKind::TypeOutlives(outlives)) => {
                self.0.register_region_obligation_with_cause(
                    outlives.0,
                    outlives.1,
                    &ObligationCause::dummy_with_span(span),
                );

                Some(HasChanged::No)
            }
            ty::PredicateKind::Clause(ty::ClauseKind::Trait(trait_pred)) => {
                match self.0.tcx.as_lang_item(trait_pred.def_id()) {
                    Some(LangItem::Sized)
                        if trait_pred.self_ty().is_trivially_sized(self.0.tcx) =>
                    {
                        Some(HasChanged::No)
                    }
                    Some(LangItem::Copy | LangItem::Clone)
                        if trait_pred.self_ty().is_trivially_pure_clone_copy() =>
                    {
                        Some(HasChanged::No)
                    }
                    _ => None,
                }
            }
            _ => None,
        }
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

    fn evaluate_const(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        uv: ty::UnevaluatedConst<'tcx>,
    ) -> Option<ty::Const<'tcx>> {
        let ct = ty::Const::new_unevaluated(self.tcx, uv);

        match crate::traits::try_evaluate_const(&self.0, ct, param_env) {
            Ok(ct) => Some(ct),
            Err(EvaluateConstErr::EvaluationFailure(e)) => Some(ty::Const::new_error(self.tcx, e)),
            Err(
                EvaluateConstErr::InvalidConstParamTy(_) | EvaluateConstErr::HasGenericsOrInfers,
            ) => None,
        }
    }

    fn well_formed_goals(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        term: ty::Term<'tcx>,
    ) -> Option<Vec<Goal<'tcx, ty::Predicate<'tcx>>>> {
        crate::traits::wf::unnormalized_obligations(
            &self.0,
            param_env,
            term,
            DUMMY_SP,
            CRATE_DEF_ID,
        )
        .map(|obligations| obligations.into_iter().map(|obligation| obligation.as_goal()).collect())
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
        kind: CanonicalVarKind<'tcx>,
        span: Span,
        universe_map: impl Fn(ty::UniverseIndex) -> ty::UniverseIndex,
    ) -> ty::GenericArg<'tcx> {
        self.0.instantiate_canonical_var(span, kind, universe_map)
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

    fn fetch_eligible_assoc_item(
        &self,
        goal_trait_ref: ty::TraitRef<'tcx>,
        trait_assoc_def_id: DefId,
        impl_def_id: DefId,
    ) -> Result<Option<DefId>, ErrorGuaranteed> {
        let node_item = specialization_graph::assoc_def(self.tcx, impl_def_id, trait_assoc_def_id)?;

        let eligible = if node_item.is_final() {
            // Non-specializable items are always projectable.
            true
        } else {
            // Only reveal a specializable default if we're past type-checking
            // and the obligation is monomorphic, otherwise passes such as
            // transmute checking and polymorphic MIR optimizations could
            // get a result which isn't correct for all monomorphizations.
            match self.typing_mode() {
                TypingMode::Coherence
                | TypingMode::Analysis { .. }
                | TypingMode::Borrowck { .. }
                | TypingMode::PostBorrowckAnalysis { .. } => false,
                TypingMode::PostAnalysis => {
                    let poly_trait_ref = self.resolve_vars_if_possible(goal_trait_ref);
                    !poly_trait_ref.still_further_specializable()
                }
            }
        };

        // FIXME: Check for defaultness here may cause diagnostics problems.
        if eligible { Ok(Some(node_item.item.def_id)) } else { Ok(None) }
    }

    // FIXME: This actually should destructure the `Result` we get from transmutability and
    // register candidates. We probably need to register >1 since we may have an OR of ANDs.
    fn is_transmutable(
        &self,
        dst: Ty<'tcx>,
        src: Ty<'tcx>,
        assume: ty::Const<'tcx>,
    ) -> Result<Certainty, NoSolution> {
        // Erase regions because we compute layouts in `rustc_transmute`,
        // which will ICE for region vars.
        let (dst, src) = self.tcx.erase_regions((dst, src));

        let Some(assume) = rustc_transmute::Assume::from_const(self.tcx, assume) else {
            return Err(NoSolution);
        };

        // FIXME(transmutability): This really should be returning nested goals for `Answer::If*`
        match rustc_transmute::TransmuteTypeEnv::new(self.0.tcx)
            .is_transmutable(rustc_transmute::Types { src, dst }, assume)
        {
            rustc_transmute::Answer::Yes => Ok(Certainty::Yes),
            rustc_transmute::Answer::No(_) | rustc_transmute::Answer::If(_) => Err(NoSolution),
        }
    }
}
