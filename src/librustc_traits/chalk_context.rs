// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use chalk_engine::fallible::Fallible as ChalkEngineFallible;
use chalk_engine::{context, hh::HhGoal, DelayedLiteral, ExClause};
use rustc::infer::canonical::{
    Canonical, CanonicalVarValues, Canonicalize, QueryRegionConstraint, QueryResult,
};
use rustc::infer::{InferCtxt, InferOk, LateBoundRegionConversionTime};
use rustc::traits::{
    WellFormed,
    FromEnv,
    DomainGoal,
    ExClauseFold,
    ExClauseLift,
    Goal,
    ProgramClause,
    QuantifierKind
};
use rustc::ty::fold::{TypeFoldable, TypeFolder, TypeVisitor};
use rustc::ty::subst::Kind;
use rustc::ty::{self, TyCtxt};

use std::fmt::{self, Debug};
use std::marker::PhantomData;

use syntax_pos::DUMMY_SP;

#[derive(Copy, Clone, Debug)]
crate struct ChalkArenas<'gcx> {
    _phantom: PhantomData<&'gcx ()>,
}

#[derive(Copy, Clone)]
crate struct ChalkContext<'cx, 'gcx: 'cx> {
    _arenas: ChalkArenas<'gcx>,
    _tcx: TyCtxt<'cx, 'gcx, 'gcx>,
}

#[derive(Copy, Clone)]
crate struct ChalkInferenceContext<'cx, 'gcx: 'tcx, 'tcx: 'cx> {
    infcx: &'cx InferCtxt<'cx, 'gcx, 'tcx>,
}

#[derive(Copy, Clone, Debug)]
crate struct UniverseMap;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
crate struct ConstrainedSubst<'tcx> {
    subst: CanonicalVarValues<'tcx>,
    constraints: Vec<QueryRegionConstraint<'tcx>>,
}

BraceStructTypeFoldableImpl! {
    impl<'tcx> TypeFoldable<'tcx> for ConstrainedSubst<'tcx> {
        subst, constraints
    }
}

impl context::Context for ChalkArenas<'tcx> {
    type CanonicalExClause = Canonical<'tcx, ExClause<Self>>;

    type CanonicalGoalInEnvironment = Canonical<'tcx, ty::ParamEnvAnd<'tcx, Goal<'tcx>>>;

    // u-canonicalization not yet implemented
    type UCanonicalGoalInEnvironment = Canonical<'tcx, ty::ParamEnvAnd<'tcx, Goal<'tcx>>>;

    type CanonicalConstrainedSubst = Canonical<'tcx, ConstrainedSubst<'tcx>>;

    // u-canonicalization not yet implemented
    type UniverseMap = UniverseMap;

    type Solution = Canonical<'tcx, QueryResult<'tcx, ()>>;

    type InferenceNormalizedSubst = CanonicalVarValues<'tcx>;

    type GoalInEnvironment = ty::ParamEnvAnd<'tcx, Goal<'tcx>>;

    type RegionConstraint = QueryRegionConstraint<'tcx>;

    type Substitution = CanonicalVarValues<'tcx>;

    type Environment = ty::ParamEnv<'tcx>;

    type Goal = Goal<'tcx>;

    type DomainGoal = DomainGoal<'tcx>;

    type BindersGoal = ty::Binder<&'tcx Goal<'tcx>>;

    type Parameter = Kind<'tcx>;

    type ProgramClause = ProgramClause<'tcx>;

    type ProgramClauses = Vec<ProgramClause<'tcx>>;

    type UnificationResult = InferOk<'tcx, ()>;

    fn into_goal(domain_goal: DomainGoal<'tcx>) -> Goal<'tcx> {
        Goal::DomainGoal(domain_goal)
    }

    fn cannot_prove() -> Goal<'tcx> {
        Goal::CannotProve
    }

    fn goal_in_environment(
        env: &ty::ParamEnv<'tcx>,
        goal: Goal<'tcx>,
    ) -> ty::ParamEnvAnd<'tcx, Goal<'tcx>> {
        env.and(goal)
    }
}

impl context::AggregateOps<ChalkArenas<'gcx>> for ChalkContext<'cx, 'gcx> {
    fn make_solution(
        &self,
        _root_goal: &Canonical<'gcx, ty::ParamEnvAnd<'gcx, Goal<'gcx>>>,
        _simplified_answers: impl context::AnswerStream<ChalkArenas<'gcx>>,
    ) -> Option<Canonical<'gcx, QueryResult<'gcx, ()>>> {
        unimplemented!()
    }
}

impl context::ContextOps<ChalkArenas<'gcx>> for ChalkContext<'cx, 'gcx> {
    /// True if this is a coinductive goal -- e.g., proving an auto trait.
    fn is_coinductive(&self, _goal: &Canonical<'gcx, ty::ParamEnvAnd<'gcx, Goal<'gcx>>>) -> bool {
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
        _arg: &Canonical<'gcx, ty::ParamEnvAnd<'gcx, Goal<'gcx>>>,
        _op: impl context::WithInstantiatedUCanonicalGoal<ChalkArenas<'gcx>, Output = R>,
    ) -> R {
        unimplemented!()
    }

    fn instantiate_ex_clause<R>(
        &self,
        _num_universes: usize,
        _canonical_ex_clause: &Canonical<'gcx, ChalkExClause<'gcx>>,
        _op: impl context::WithInstantiatedExClause<ChalkArenas<'gcx>, Output = R>,
    ) -> R {
        unimplemented!()
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
        u_canon: &'a Canonical<'gcx, ty::ParamEnvAnd<'gcx, Goal<'gcx>>>,
    ) -> &'a Canonical<'gcx, ty::ParamEnvAnd<'gcx, Goal<'gcx>>> {
        u_canon
    }

    fn is_trivial_substitution(
        _u_canon: &Canonical<'gcx, ty::ParamEnvAnd<'gcx, Goal<'gcx>>>,
        _canonical_subst: &Canonical<'gcx, ConstrainedSubst<'gcx>>,
    ) -> bool {
        unimplemented!()
    }

    fn num_universes(_: &Canonical<'gcx, ty::ParamEnvAnd<'gcx, Goal<'gcx>>>) -> usize {
        0 // FIXME
    }

    /// Convert a goal G *from* the canonical universes *into* our
    /// local universes. This will yield a goal G' that is the same
    /// but for the universes of universally quantified names.
    fn map_goal_from_canonical(
        _map: &UniverseMap,
        value: &Canonical<'gcx, ty::ParamEnvAnd<'gcx, Goal<'gcx>>>,
    ) -> Canonical<'gcx, ty::ParamEnvAnd<'gcx, Goal<'gcx>>> {
        *value // FIXME universe maps not implemented yet
    }

    fn map_subst_from_canonical(
        _map: &UniverseMap,
        value: &Canonical<'gcx, ConstrainedSubst<'gcx>>,
    ) -> Canonical<'gcx, ConstrainedSubst<'gcx>> {
        value.clone() // FIXME universe maps not implemented yet
    }
}

//impl context::UCanonicalGoalInEnvironment<ChalkContext<'cx, 'gcx>>
//    for Canonical<'gcx, ty::ParamEnvAnd<'gcx, Goal<'gcx>>>
//{
//    fn canonical(&self) -> &Canonical<'gcx, ty::ParamEnvAnd<'gcx, Goal<'gcx>>> {
//        self
//    }
//
//    fn is_trivial_substitution(
//        &self,
//        canonical_subst: &Canonical<'tcx, ConstrainedSubst<'tcx>>,
//    ) -> bool {
//        let subst = &canonical_subst.value.subst;
//        assert_eq!(self.canonical.variables.len(), subst.var_values.len());
//        subst
//            .var_values
//            .iter_enumerated()
//            .all(|(cvar, kind)| match kind.unpack() {
//                Kind::Lifetime(r) => match r {
//                    ty::ReCanonical(cvar1) => cvar == cvar1,
//                    _ => false,
//                },
//                Kind::Type(ty) => match ty.sty {
//                    ty::TyInfer(ty::InferTy::CanonicalTy(cvar1)) => cvar == cvar1,
//                    _ => false,
//                },
//            })
//    }
//
//    fn num_universes(&self) -> usize {
//        0 // FIXME
//    }
//}

impl context::InferenceTable<ChalkArenas<'gcx>, ChalkArenas<'tcx>>
    for ChalkInferenceContext<'cx, 'gcx, 'tcx>
{
    fn into_hh_goal(&mut self, goal: Goal<'tcx>) -> ChalkHhGoal<'tcx> {
        match goal {
            Goal::Implies(..) => panic!("FIXME rust-lang-nursery/chalk#94"),
            Goal::And(left, right) => HhGoal::And(*left, *right),
            Goal::Not(subgoal) => HhGoal::Not(*subgoal),
            Goal::DomainGoal(d) => HhGoal::DomainGoal(d),
            Goal::Quantified(QuantifierKind::Universal, binder) => HhGoal::ForAll(binder),
            Goal::Quantified(QuantifierKind::Existential, binder) => HhGoal::Exists(binder),
            Goal::CannotProve => HhGoal::CannotProve,
        }
    }

    fn add_clauses(
        &mut self,
        _env: &ty::ParamEnv<'tcx>,
        _clauses: Vec<ProgramClause<'tcx>>,
    ) -> ty::ParamEnv<'tcx> {
        panic!("FIXME no method to add clauses to ParamEnv yet")
    }
}

impl context::ResolventOps<ChalkArenas<'gcx>, ChalkArenas<'tcx>>
    for ChalkInferenceContext<'cx, 'gcx, 'tcx>
{
    fn resolvent_clause(
        &mut self,
        _environment: &ty::ParamEnv<'tcx>,
        _goal: &DomainGoal<'tcx>,
        _subst: &CanonicalVarValues<'tcx>,
        _clause: &ProgramClause<'tcx>,
    ) -> chalk_engine::fallible::Fallible<Canonical<'gcx, ChalkExClause<'gcx>>> {
        panic!()
    }

    fn apply_answer_subst(
        &mut self,
        _ex_clause: ChalkExClause<'tcx>,
        _selected_goal: &ty::ParamEnvAnd<'tcx, Goal<'tcx>>,
        _answer_table_goal: &Canonical<'gcx, ty::ParamEnvAnd<'gcx, Goal<'gcx>>>,
        _canonical_answer_subst: &Canonical<'gcx, ConstrainedSubst<'gcx>>,
    ) -> chalk_engine::fallible::Fallible<ChalkExClause<'tcx>> {
        panic!()
    }
}

impl context::TruncateOps<ChalkArenas<'gcx>, ChalkArenas<'tcx>>
    for ChalkInferenceContext<'cx, 'gcx, 'tcx>
{
    fn truncate_goal(
        &mut self,
        subgoal: &ty::ParamEnvAnd<'tcx, Goal<'tcx>>,
    ) -> Option<ty::ParamEnvAnd<'tcx, Goal<'tcx>>> {
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
        _environment: &ty::ParamEnv<'tcx>,
        goal: &DomainGoal<'tcx>,
    ) -> Vec<ProgramClause<'tcx>> {
        use rustc::traits::WhereClause::*;

        match goal {
            DomainGoal::Holds(Implemented(_trait_predicate)) => {
                // These come from:
                //
                // - Trait definitions (implied bounds)
                // - Implementations of the trait itself
                panic!()
            }

            DomainGoal::Holds(ProjectionEq(_projection_predicate)) => {
                // These come from:
                panic!()
            }

            DomainGoal::Holds(RegionOutlives(_region_outlives)) => {
                panic!()
            }

            DomainGoal::Holds(TypeOutlives(_type_outlives)) => {
                panic!()
            }

            DomainGoal::WellFormed(WellFormed::Trait(_trait_predicate)) => {
                // These come from -- the trait decl.
                panic!()
            }

            DomainGoal::WellFormed(WellFormed::Ty(_ty)) => panic!(),

            DomainGoal::FromEnv(FromEnv::Trait(_trait_predicate)) => panic!(),

            DomainGoal::FromEnv(FromEnv::Ty(_ty)) => panic!(),

            DomainGoal::Normalize(_) => panic!(),
        }
    }

    fn instantiate_binders_universally(
        &mut self,
        _arg: &ty::Binder<&'tcx Goal<'tcx>>,
    ) -> Goal<'tcx> {
        panic!("FIXME -- universal instantiation needs sgrif's branch")
    }

    fn instantiate_binders_existentially(
        &mut self,
        arg: &ty::Binder<&'tcx Goal<'tcx>>,
    ) -> Goal<'tcx> {
        let (value, _map) = self.infcx.replace_late_bound_regions_with_fresh_var(
            DUMMY_SP,
            LateBoundRegionConversionTime::HigherRankedType,
            arg,
        );
        *value
    }

    fn debug_ex_clause(&mut self, value: &'v ChalkExClause<'tcx>) -> Box<dyn Debug + 'v> {
        let string = format!("{:?}", self.infcx.resolve_type_vars_if_possible(value));
        Box::new(string)
    }

    fn canonicalize_goal(
        &mut self,
        value: &ty::ParamEnvAnd<'tcx, Goal<'tcx>>,
    ) -> Canonical<'gcx, ty::ParamEnvAnd<'gcx, Goal<'gcx>>> {
        self.infcx.canonicalize_query(value).0
    }

    fn canonicalize_ex_clause(
        &mut self,
        value: &ChalkExClause<'tcx>,
    ) -> Canonical<'gcx, ChalkExClause<'gcx>> {
        self.infcx.canonicalize_response(value).0
    }

    fn canonicalize_constrained_subst(
        &mut self,
        subst: CanonicalVarValues<'tcx>,
        constraints: Vec<QueryRegionConstraint<'tcx>>,
    ) -> Canonical<'gcx, ConstrainedSubst<'gcx>> {
        self.infcx
            .canonicalize_response(&ConstrainedSubst { subst, constraints })
            .0
    }

    fn u_canonicalize_goal(
        &mut self,
        value: &Canonical<'gcx, ty::ParamEnvAnd<'gcx, Goal<'gcx>>>,
    ) -> (
        Canonical<'gcx, ty::ParamEnvAnd<'gcx, Goal<'gcx>>>,
        UniverseMap,
    ) {
        (value.clone(), UniverseMap)
    }

    fn invert_goal(
        &mut self,
        _value: &ty::ParamEnvAnd<'tcx, Goal<'tcx>>,
    ) -> Option<ty::ParamEnvAnd<'tcx, Goal<'tcx>>> {
        panic!("goal inversion not yet implemented")
    }

    fn unify_parameters(
        &mut self,
        _environment: &ty::ParamEnv<'tcx>,
        _a: &Kind<'tcx>,
        _b: &Kind<'tcx>,
    ) -> ChalkEngineFallible<InferOk<'tcx, ()>> {
        panic!()
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

    fn into_ex_clause(&mut self, _result: InferOk<'tcx, ()>, _ex_clause: &mut ChalkExClause<'tcx>) {
        panic!("TBD")
    }
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

impl<'gcx: 'tcx, 'tcx> Canonicalize<'gcx, 'tcx> for ConstrainedSubst<'tcx> {
    type Canonicalized = Canonical<'gcx, ConstrainedSubst<'gcx>>;

    fn intern(
        _gcx: TyCtxt<'_, 'gcx, 'gcx>,
        value: Canonical<'gcx, ConstrainedSubst<'gcx>>,
    ) -> Self::Canonicalized {
        value
    }
}
