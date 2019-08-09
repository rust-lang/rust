use rustc::infer::nll_relate::{TypeRelating, TypeRelatingDelegate, NormalizationStrategy};
use rustc::infer::{InferCtxt, RegionVariableOrigin};
use rustc::traits::{DomainGoal, Goal, Environment, InEnvironment};
use rustc::ty::relate::{Relate, TypeRelation, RelateResult};
use rustc::ty;
use syntax_pos::DUMMY_SP;

crate struct UnificationResult<'tcx> {
    crate goals: Vec<InEnvironment<'tcx, Goal<'tcx>>>,
    crate constraints: Vec<super::RegionConstraint<'tcx>>,
}

crate fn unify<'me, 'tcx, T: Relate<'tcx>>(
    infcx: &'me InferCtxt<'me, 'tcx>,
    environment: Environment<'tcx>,
    variance: ty::Variance,
    a: &T,
    b: &T,
) -> RelateResult<'tcx, UnificationResult<'tcx>> {
    debug!("unify(
        a = {:?},
        b = {:?},
        environment = {:?},
    )", a, b, environment);

    let mut delegate = ChalkTypeRelatingDelegate::new(
        infcx,
        environment
    );

    TypeRelating::new(
        infcx,
        &mut delegate,
        variance
    ).relate(a, b)?;

    debug!("unify: goals = {:?}, constraints = {:?}", delegate.goals, delegate.constraints);

    Ok(UnificationResult {
        goals: delegate.goals,
        constraints: delegate.constraints,
    })
}

struct ChalkTypeRelatingDelegate<'me, 'tcx> {
    infcx: &'me InferCtxt<'me, 'tcx>,
    environment: Environment<'tcx>,
    goals: Vec<InEnvironment<'tcx, Goal<'tcx>>>,
    constraints: Vec<super::RegionConstraint<'tcx>>,
}

impl ChalkTypeRelatingDelegate<'me, 'tcx> {
    fn new(infcx: &'me InferCtxt<'me, 'tcx>, environment: Environment<'tcx>) -> Self {
        Self {
            infcx,
            environment,
            goals: Vec::new(),
            constraints: Vec::new(),
        }
    }
}

impl TypeRelatingDelegate<'tcx> for &mut ChalkTypeRelatingDelegate<'_, 'tcx> {
    fn create_next_universe(&mut self) -> ty::UniverseIndex {
        self.infcx.create_next_universe()
    }

    fn next_existential_region_var(&mut self) -> ty::Region<'tcx> {
        self.infcx.next_region_var(RegionVariableOrigin::MiscVariable(DUMMY_SP))
    }

    fn next_placeholder_region(
        &mut self,
        placeholder: ty::PlaceholderRegion
    ) -> ty::Region<'tcx> {
        self.infcx.tcx.mk_region(ty::RePlaceholder(placeholder))
    }

    fn generalize_existential(&mut self, universe: ty::UniverseIndex) -> ty::Region<'tcx> {
        self.infcx.next_region_var_in_universe(
            RegionVariableOrigin::MiscVariable(DUMMY_SP),
            universe
        )
    }

    fn push_outlives(&mut self, sup: ty::Region<'tcx>, sub: ty::Region<'tcx>) {
        self.constraints.push(ty::OutlivesPredicate(sup.into(), sub));
    }

    fn push_domain_goal(&mut self, domain_goal: DomainGoal<'tcx>) {
        let goal = self.environment.with(
            self.infcx.tcx.mk_goal(domain_goal.into_goal())
        );
        self.goals.push(goal);
    }

    fn normalization() -> NormalizationStrategy {
        NormalizationStrategy::Lazy
    }

    fn forbid_inference_vars() -> bool {
        false
    }
}
