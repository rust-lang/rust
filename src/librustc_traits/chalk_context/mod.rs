mod program_clauses;
mod resolvent_ops;
mod unify;

use chalk_engine::fallible::{Fallible, NoSolution};
use chalk_engine::{
    context,
    hh::HhGoal,
    DelayedLiteral,
    Literal,
    ExClause
};
use rustc::infer::{InferCtxt, LateBoundRegionConversionTime};
use rustc::infer::canonical::{
    Canonical,
    CanonicalVarValues,
    OriginalQueryValues,
    QueryResponse,
    Certainty,
};
use rustc::traits::{
    DomainGoal,
    ExClauseFold,
    ExClauseLift,
    Goal,
    GoalKind,
    Clause,
    QuantifierKind,
    Environment,
    InEnvironment,
};
use rustc::ty::{self, TyCtxt};
use rustc::ty::fold::{TypeFoldable, TypeFolder, TypeVisitor};
use rustc::ty::subst::{Kind, UnpackedKind};
use syntax_pos::DUMMY_SP;

use std::fmt::{self, Debug};
use std::marker::PhantomData;

use self::unify::*;

#[derive(Copy, Clone, Debug)]
crate struct ChalkArenas<'gcx> {
    _phantom: PhantomData<&'gcx ()>,
}

#[derive(Copy, Clone)]
crate struct ChalkContext<'cx, 'gcx: 'cx> {
    _arenas: ChalkArenas<'gcx>,
    tcx: TyCtxt<'cx, 'gcx, 'gcx>,
}

#[derive(Copy, Clone)]
crate struct ChalkInferenceContext<'cx, 'gcx: 'tcx, 'tcx: 'cx> {
    infcx: &'cx InferCtxt<'cx, 'gcx, 'tcx>,
}

#[derive(Copy, Clone, Debug)]
crate struct UniverseMap;

crate type RegionConstraint<'tcx> = ty::OutlivesPredicate<Kind<'tcx>, ty::Region<'tcx>>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
crate struct ConstrainedSubst<'tcx> {
    subst: CanonicalVarValues<'tcx>,
    constraints: Vec<RegionConstraint<'tcx>>,
}

BraceStructTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for ConstrainedSubst<'tcx> {
        subst, constraints
    }
}

impl context::Context for ChalkArenas<'tcx> {
    type CanonicalExClause = Canonical<'tcx, ChalkExClause<'tcx>>;

    type CanonicalGoalInEnvironment = Canonical<'tcx, InEnvironment<'tcx, Goal<'tcx>>>;

    // u-canonicalization not yet implemented
    type UCanonicalGoalInEnvironment = Canonical<'tcx, InEnvironment<'tcx, Goal<'tcx>>>;

    type CanonicalConstrainedSubst = Canonical<'tcx, ConstrainedSubst<'tcx>>;

    // u-canonicalization not yet implemented
    type UniverseMap = UniverseMap;

    type Solution = Canonical<'tcx, QueryResponse<'tcx, ()>>;

    type InferenceNormalizedSubst = CanonicalVarValues<'tcx>;

    type GoalInEnvironment = InEnvironment<'tcx, Goal<'tcx>>;

    type RegionConstraint = RegionConstraint<'tcx>;

    type Substitution = CanonicalVarValues<'tcx>;

    type Environment = Environment<'tcx>;

    type Goal = Goal<'tcx>;

    type DomainGoal = DomainGoal<'tcx>;

    type BindersGoal = ty::Binder<Goal<'tcx>>;

    type Parameter = Kind<'tcx>;

    type ProgramClause = Clause<'tcx>;

    type ProgramClauses = Vec<Clause<'tcx>>;

    type UnificationResult = UnificationResult<'tcx>;

    fn goal_in_environment(
        env: &Environment<'tcx>,
        goal: Goal<'tcx>,
    ) -> InEnvironment<'tcx, Goal<'tcx>> {
        env.with(goal)
    }
}

impl context::AggregateOps<ChalkArenas<'gcx>> for ChalkContext<'cx, 'gcx> {
    fn make_solution(
        &self,
        _root_goal: &Canonical<'gcx, InEnvironment<'gcx, Goal<'gcx>>>,
        mut simplified_answers: impl context::AnswerStream<ChalkArenas<'gcx>>,
    ) -> Option<Canonical<'gcx, QueryResponse<'gcx, ()>>> {
        use chalk_engine::SimplifiedAnswer;

        if simplified_answers.peek_answer().is_none() {
            return None;
        }

        let SimplifiedAnswer { subst, ambiguous } = simplified_answers
            .next_answer()
            .unwrap();

        let ambiguous = simplified_answers.peek_answer().is_some() || ambiguous;

        Some(subst.unchecked_map(|subst| {
            QueryResponse {
                var_values: subst.subst,
                region_constraints: subst.constraints
                    .into_iter()
                    .map(|c| ty::Binder::bind(c))
                    .collect(),
                certainty: match ambiguous {
                    true => Certainty::Ambiguous,
                    false => Certainty::Proven,
                },
                value: (),
            }
        }))
    }
}

impl context::ContextOps<ChalkArenas<'gcx>> for ChalkContext<'cx, 'gcx> {
    /// True if this is a coinductive goal -- e.g., proving an auto trait.
    fn is_coinductive(
        &self,
        _goal: &Canonical<'gcx, InEnvironment<'gcx, Goal<'gcx>>>
    ) -> bool {
        unimplemented!()
    }

    /// Create an inference table for processing a new goal and instantiate that goal
    /// in that context, returning "all the pieces".
    ///
    /// More specifically: given a u-canonical goal `arg`, creates a
    /// new inference table `T` and populates it with the universes
    /// found in `arg`. Then, creates a substitution `S` that maps
    /// each bound variable in `arg` to a fresh inference variable
    /// from T. Returns:
    ///
    /// - the table `T`
    /// - the substitution `S`
    /// - the environment and goal found by substitution `S` into `arg`
    fn instantiate_ucanonical_goal<R>(
        &self,
        arg: &Canonical<'gcx, InEnvironment<'gcx, Goal<'gcx>>>,
        op: impl context::WithInstantiatedUCanonicalGoal<ChalkArenas<'gcx>, Output = R>,
    ) -> R {
        self.tcx.infer_ctxt().enter_with_canonical(DUMMY_SP, arg, |ref infcx, arg, subst| {
            let chalk_infcx = &mut ChalkInferenceContext {
                infcx,
            };
            op.with(chalk_infcx, subst, arg.environment, arg.goal)
        })
    }

    fn instantiate_ex_clause<R>(
        &self,
        _num_universes: usize,
        arg: &Canonical<'gcx, ChalkExClause<'gcx>>,
        op: impl context::WithInstantiatedExClause<ChalkArenas<'gcx>, Output = R>,
    ) -> R {
        self.tcx.infer_ctxt().enter_with_canonical(DUMMY_SP, &arg.upcast(), |ref infcx, arg, _| {
            let chalk_infcx = &mut ChalkInferenceContext {
                infcx,
            };
            op.with(chalk_infcx,arg)
        })
    }

    /// True if this solution has no region constraints.
    fn empty_constraints(ccs: &Canonical<'gcx, ConstrainedSubst<'gcx>>) -> bool {
        ccs.value.constraints.is_empty()
    }

    fn inference_normalized_subst_from_ex_clause(
        canon_ex_clause: &'a Canonical<'gcx, ChalkExClause<'gcx>>,
    ) -> &'a CanonicalVarValues<'gcx> {
        &canon_ex_clause.value.subst
    }

    fn inference_normalized_subst_from_subst(
        canon_subst: &'a Canonical<'gcx, ConstrainedSubst<'gcx>>,
    ) -> &'a CanonicalVarValues<'gcx> {
        &canon_subst.value.subst
    }

    fn canonical(
        u_canon: &'a Canonical<'gcx, InEnvironment<'gcx, Goal<'gcx>>>,
    ) -> &'a Canonical<'gcx, InEnvironment<'gcx, Goal<'gcx>>> {
        u_canon
    }

    fn is_trivial_substitution(
        u_canon: &Canonical<'gcx, InEnvironment<'gcx, Goal<'gcx>>>,
        canonical_subst: &Canonical<'gcx, ConstrainedSubst<'gcx>>,
    ) -> bool {
        let subst = &canonical_subst.value.subst;
        assert_eq!(u_canon.variables.len(), subst.var_values.len());
        subst.var_values
            .iter_enumerated()
            .all(|(cvar, kind)| match kind.unpack() {
                UnpackedKind::Lifetime(r) => match r {
                    &ty::ReLateBound(debruijn, br) => {
                        debug_assert_eq!(debruijn, ty::INNERMOST);
                        cvar == br.assert_bound_var()
                    }
                    _ => false,
                },
                UnpackedKind::Type(ty) => match ty.sty {
                    ty::Bound(debruijn, bound_ty) => {
                        debug_assert_eq!(debruijn, ty::INNERMOST);
                        cvar == bound_ty.var
                    }
                    _ => false,
                },
            })
    }

    fn num_universes(canon: &Canonical<'gcx, InEnvironment<'gcx, Goal<'gcx>>>) -> usize {
        canon.max_universe.index() + 1
    }

    /// Convert a goal G *from* the canonical universes *into* our
    /// local universes. This will yield a goal G' that is the same
    /// but for the universes of universally quantified names.
    fn map_goal_from_canonical(
        _map: &UniverseMap,
        value: &Canonical<'gcx, InEnvironment<'gcx, Goal<'gcx>>>,
    ) -> Canonical<'gcx, InEnvironment<'gcx, Goal<'gcx>>> {
        *value // FIXME universe maps not implemented yet
    }

    fn map_subst_from_canonical(
        _map: &UniverseMap,
        value: &Canonical<'gcx, ConstrainedSubst<'gcx>>,
    ) -> Canonical<'gcx, ConstrainedSubst<'gcx>> {
        value.clone() // FIXME universe maps not implemented yet
    }
}

impl context::InferenceTable<ChalkArenas<'gcx>, ChalkArenas<'tcx>>
    for ChalkInferenceContext<'cx, 'gcx, 'tcx>
{
    fn into_goal(&self, domain_goal: DomainGoal<'tcx>) -> Goal<'tcx> {
        self.infcx.tcx.mk_goal(GoalKind::DomainGoal(domain_goal))
    }

    fn cannot_prove(&self) -> Goal<'tcx> {
        self.infcx.tcx.mk_goal(GoalKind::CannotProve)
    }

    fn into_hh_goal(&mut self, goal: Goal<'tcx>) -> ChalkHhGoal<'tcx> {
        match *goal {
            GoalKind::Implies(hypotheses, goal) => HhGoal::Implies(
                hypotheses.iter().cloned().collect(),
                goal
            ),
            GoalKind::And(left, right) => HhGoal::And(left, right),
            GoalKind::Not(subgoal) => HhGoal::Not(subgoal),
            GoalKind::DomainGoal(d) => HhGoal::DomainGoal(d),
            GoalKind::Quantified(QuantifierKind::Universal, binder) => HhGoal::ForAll(binder),
            GoalKind::Quantified(QuantifierKind::Existential, binder) => HhGoal::Exists(binder),
            GoalKind::CannotProve => HhGoal::CannotProve,
        }
    }

    fn add_clauses(
        &mut self,
        env: &Environment<'tcx>,
        clauses: Vec<Clause<'tcx>>,
    ) -> Environment<'tcx> {
        Environment {
            clauses: self.infcx.tcx.mk_clauses(
                env.clauses.iter().cloned().chain(clauses.into_iter())
            )
        }
    }
}

impl context::TruncateOps<ChalkArenas<'gcx>, ChalkArenas<'tcx>>
    for ChalkInferenceContext<'cx, 'gcx, 'tcx>
{
    fn truncate_goal(
        &mut self,
        subgoal: &InEnvironment<'tcx, Goal<'tcx>>,
    ) -> Option<InEnvironment<'tcx, Goal<'tcx>>> {
        Some(*subgoal) // FIXME we should truncate at some point!
    }

    fn truncate_answer(
        &mut self,
        subst: &CanonicalVarValues<'tcx>,
    ) -> Option<CanonicalVarValues<'tcx>> {
        Some(subst.clone()) // FIXME we should truncate at some point!
    }
}

impl context::UnificationOps<ChalkArenas<'gcx>, ChalkArenas<'tcx>>
    for ChalkInferenceContext<'cx, 'gcx, 'tcx>
{
    fn program_clauses(
        &self,
        environment: &Environment<'tcx>,
        goal: &DomainGoal<'tcx>,
    ) -> Vec<Clause<'tcx>> {
        self.program_clauses_impl(environment, goal)
    }

    fn instantiate_binders_universally(
        &mut self,
        arg: &ty::Binder<Goal<'tcx>>,
    ) -> Goal<'tcx> {
        self.infcx.replace_bound_vars_with_placeholders(arg).0
    }

    fn instantiate_binders_existentially(
        &mut self,
        arg: &ty::Binder<Goal<'tcx>>,
    ) -> Goal<'tcx> {
        self.infcx.replace_bound_vars_with_fresh_vars(
            DUMMY_SP,
            LateBoundRegionConversionTime::HigherRankedType,
            arg
        ).0
    }

    fn debug_ex_clause(&mut self, value: &'v ChalkExClause<'tcx>) -> Box<dyn Debug + 'v> {
        let string = format!("{:?}", self.infcx.resolve_type_vars_if_possible(value));
        Box::new(string)
    }

    fn canonicalize_goal(
        &mut self,
        value: &InEnvironment<'tcx, Goal<'tcx>>,
    ) -> Canonical<'gcx, InEnvironment<'gcx, Goal<'gcx>>> {
        let mut _orig_values = OriginalQueryValues::default();
        self.infcx.canonicalize_query(value, &mut _orig_values)
    }

    fn canonicalize_ex_clause(
        &mut self,
        value: &ChalkExClause<'tcx>,
    ) -> Canonical<'gcx, ChalkExClause<'gcx>> {
        self.infcx.canonicalize_response(value)
    }

    fn canonicalize_constrained_subst(
        &mut self,
        subst: CanonicalVarValues<'tcx>,
        constraints: Vec<RegionConstraint<'tcx>>,
    ) -> Canonical<'gcx, ConstrainedSubst<'gcx>> {
        self.infcx.canonicalize_response(&ConstrainedSubst { subst, constraints })
    }

    fn u_canonicalize_goal(
        &mut self,
        value: &Canonical<'gcx, InEnvironment<'gcx, Goal<'gcx>>>,
    ) -> (
        Canonical<'gcx, InEnvironment<'gcx, Goal<'gcx>>>,
        UniverseMap,
    ) {
        (value.clone(), UniverseMap)
    }

    fn invert_goal(
        &mut self,
        _value: &InEnvironment<'tcx, Goal<'tcx>>,
    ) -> Option<InEnvironment<'tcx, Goal<'tcx>>> {
        panic!("goal inversion not yet implemented")
    }

    fn unify_parameters(
        &mut self,
        environment: &Environment<'tcx>,
        a: &Kind<'tcx>,
        b: &Kind<'tcx>,
    ) -> Fallible<UnificationResult<'tcx>> {
        self.infcx.commit_if_ok(|_| {
            unify(self.infcx, *environment, a, b).map_err(|_| NoSolution)
        })
    }

    fn sink_answer_subset(
        &self,
        value: &Canonical<'gcx, ConstrainedSubst<'gcx>>,
    ) -> Canonical<'tcx, ConstrainedSubst<'tcx>> {
        value.clone()
    }

    fn lift_delayed_literal(
        &self,
        _value: DelayedLiteral<ChalkArenas<'tcx>>,
    ) -> DelayedLiteral<ChalkArenas<'gcx>> {
        panic!("lift")
    }

    fn into_ex_clause(
        &mut self,
        result: UnificationResult<'tcx>,
        ex_clause: &mut ChalkExClause<'tcx>
    ) {
        into_ex_clause(result, ex_clause);
    }
}

crate fn into_ex_clause(result: UnificationResult<'tcx>, ex_clause: &mut ChalkExClause<'tcx>) {
    ex_clause.subgoals.extend(
        result.goals.into_iter().map(Literal::Positive)
    );
    ex_clause.constraints.extend(result.constraints);
}

type ChalkHhGoal<'tcx> = HhGoal<ChalkArenas<'tcx>>;

type ChalkExClause<'tcx> = ExClause<ChalkArenas<'tcx>>;

impl Debug for ChalkContext<'cx, 'gcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ChalkContext")
    }
}

impl Debug for ChalkInferenceContext<'cx, 'gcx, 'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ChalkInferenceContext")
    }
}

impl ExClauseLift<'gcx> for ChalkArenas<'a> {
    type LiftedExClause = ChalkExClause<'gcx>;

    fn lift_ex_clause_to_tcx(
        _ex_clause: &ChalkExClause<'a>,
        _tcx: TyCtxt<'_, '_, 'tcx>,
    ) -> Option<Self::LiftedExClause> {
        panic!()
    }
}

impl ExClauseFold<'tcx> for ChalkArenas<'tcx> {
    fn fold_ex_clause_with<'gcx: 'tcx, F: TypeFolder<'gcx, 'tcx>>(
        ex_clause: &ChalkExClause<'tcx>,
        folder: &mut F,
    ) -> ChalkExClause<'tcx> {
        ExClause {
            subst: ex_clause.subst.fold_with(folder),
            delayed_literals: ex_clause.delayed_literals.fold_with(folder),
            constraints: ex_clause.constraints.fold_with(folder),
            subgoals: ex_clause.subgoals.fold_with(folder),
        }
    }

    fn visit_ex_clause_with<'gcx: 'tcx, V: TypeVisitor<'tcx>>(
        ex_clause: &ExClause<Self>,
        visitor: &mut V,
    ) -> bool {
        let ExClause {
            subst,
            delayed_literals,
            constraints,
            subgoals,
        } = ex_clause;
        subst.visit_with(visitor)
            && delayed_literals.visit_with(visitor)
            && constraints.visit_with(visitor)
            && subgoals.visit_with(visitor)
    }
}

BraceStructLiftImpl! {
    impl<'a, 'tcx> Lift<'tcx> for ConstrainedSubst<'a> {
        type Lifted = ConstrainedSubst<'tcx>;

        subst, constraints
    }
}

trait Upcast<'tcx, 'gcx: 'tcx>: 'gcx {
    type Upcasted: 'tcx;

    fn upcast(&self) -> Self::Upcasted;
}

impl<'tcx, 'gcx: 'tcx> Upcast<'tcx, 'gcx> for DelayedLiteral<ChalkArenas<'gcx>> {
    type Upcasted = DelayedLiteral<ChalkArenas<'tcx>>;

    fn upcast(&self) -> Self::Upcasted {
        match self {
            &DelayedLiteral::CannotProve(..) => DelayedLiteral::CannotProve(()),
            &DelayedLiteral::Negative(index) => DelayedLiteral::Negative(index),
            DelayedLiteral::Positive(index, subst) => DelayedLiteral::Positive(
                *index,
                subst.clone()
            ),
        }
    }
}

impl<'tcx, 'gcx: 'tcx> Upcast<'tcx, 'gcx> for Literal<ChalkArenas<'gcx>> {
    type Upcasted = Literal<ChalkArenas<'tcx>>;

    fn upcast(&self) -> Self::Upcasted {
        match self {
            &Literal::Negative(goal) => Literal::Negative(goal),
            &Literal::Positive(goal) => Literal::Positive(goal),
        }
    }
}

impl<'tcx, 'gcx: 'tcx> Upcast<'tcx, 'gcx> for ExClause<ChalkArenas<'gcx>> {
    type Upcasted = ExClause<ChalkArenas<'tcx>>;

    fn upcast(&self) -> Self::Upcasted {
        ExClause {
            subst: self.subst.clone(),
            delayed_literals: self.delayed_literals
                .iter()
                .map(|l| l.upcast())
                .collect(),
            constraints: self.constraints.clone(),
            subgoals: self.subgoals
                .iter()
                .map(|g| g.upcast())
                .collect(),
        }
    }
}

impl<'tcx, 'gcx: 'tcx, T> Upcast<'tcx, 'gcx> for Canonical<'gcx, T>
    where T: Upcast<'tcx, 'gcx>
{
    type Upcasted = Canonical<'tcx, T::Upcasted>;

    fn upcast(&self) -> Self::Upcasted {
        Canonical {
            max_universe: self.max_universe,
            value: self.value.upcast(),
            variables: self.variables,
        }
    }
}
