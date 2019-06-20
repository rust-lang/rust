use chalk_engine::fallible::{Fallible, NoSolution};
use chalk_engine::{
    context,
    Literal,
    ExClause
};
use rustc::infer::{InferCtxt, LateBoundRegionConversionTime};
use rustc::infer::canonical::{Canonical, CanonicalVarValues};
use rustc::traits::{
    DomainGoal,
    WhereClause,
    Goal,
    GoalKind,
    Clause,
    ProgramClause,
    Environment,
    InEnvironment,
};
use rustc::ty::{self, Ty, TyCtxt, InferConst};
use rustc::ty::subst::Kind;
use rustc::ty::relate::{Relate, RelateResult, TypeRelation};
use rustc::mir::interpret::ConstValue;
use syntax_pos::DUMMY_SP;

use super::{ChalkInferenceContext, ChalkArenas, ChalkExClause, ConstrainedSubst};
use super::unify::*;

impl context::ResolventOps<ChalkArenas<'tcx>, ChalkArenas<'tcx>>
    for ChalkInferenceContext<'cx, 'tcx>
{
    fn resolvent_clause(
        &mut self,
        environment: &Environment<'tcx>,
        goal: &DomainGoal<'tcx>,
        subst: &CanonicalVarValues<'tcx>,
        clause: &Clause<'tcx>,
    ) -> Fallible<Canonical<'tcx, ChalkExClause<'tcx>>> {
        use chalk_engine::context::UnificationOps;

        debug!("resolvent_clause(goal = {:?}, clause = {:?})", goal, clause);

        let result = self.infcx.probe(|_| {
            let ProgramClause {
                goal: consequence,
                hypotheses,
                ..
            } = match clause {
                Clause::Implies(program_clause) => *program_clause,
                Clause::ForAll(program_clause) => self.infcx.replace_bound_vars_with_fresh_vars(
                    DUMMY_SP,
                    LateBoundRegionConversionTime::HigherRankedType,
                    program_clause
                ).0,
            };

            let result = unify(
                self.infcx,
                *environment,
                ty::Variance::Invariant,
                goal,
                &consequence
            ).map_err(|_| NoSolution)?;

            let mut ex_clause = ExClause {
                subst: subst.clone(),
                delayed_literals: vec![],
                constraints: vec![],
                subgoals: vec![],
            };

            self.into_ex_clause(result, &mut ex_clause);

            ex_clause.subgoals.extend(
                hypotheses.iter().map(|g| match g {
                    GoalKind::Not(g) => Literal::Negative(environment.with(*g)),
                    g => Literal::Positive(environment.with(*g)),
                })
            );

            // If we have a goal of the form `T: 'a` or `'a: 'b`, then just
            // assume it is true (no subgoals) and register it as a constraint
            // instead.
            match goal {
                DomainGoal::Holds(WhereClause::RegionOutlives(pred)) => {
                    assert_eq!(ex_clause.subgoals.len(), 0);
                    ex_clause.constraints.push(ty::OutlivesPredicate(pred.0.into(), pred.1));
                }

                DomainGoal::Holds(WhereClause::TypeOutlives(pred)) => {
                    assert_eq!(ex_clause.subgoals.len(), 0);
                    ex_clause.constraints.push(ty::OutlivesPredicate(pred.0.into(), pred.1));
                }

                _ => (),
            };

            let canonical_ex_clause = self.canonicalize_ex_clause(&ex_clause);
            Ok(canonical_ex_clause)
        });

        debug!("resolvent_clause: result = {:?}", result);
        result
    }

    fn apply_answer_subst(
        &mut self,
        ex_clause: ChalkExClause<'tcx>,
        selected_goal: &InEnvironment<'tcx, Goal<'tcx>>,
        answer_table_goal: &Canonical<'tcx, InEnvironment<'tcx, Goal<'tcx>>>,
        canonical_answer_subst: &Canonical<'tcx, ConstrainedSubst<'tcx>>,
    ) -> Fallible<ChalkExClause<'tcx>> {
        debug!(
            "apply_answer_subst(ex_clause = {:?}, selected_goal = {:?})",
            self.infcx.resolve_vars_if_possible(&ex_clause),
            self.infcx.resolve_vars_if_possible(selected_goal)
        );

        let (answer_subst, _) = self.infcx.instantiate_canonical_with_fresh_inference_vars(
            DUMMY_SP,
            canonical_answer_subst
        );

        let mut substitutor = AnswerSubstitutor {
            infcx: self.infcx,
            environment: selected_goal.environment,
            answer_subst: answer_subst.subst,
            binder_index: ty::INNERMOST,
            ex_clause,
        };

        substitutor.relate(&answer_table_goal.value, &selected_goal)
            .map_err(|_| NoSolution)?;

        let mut ex_clause = substitutor.ex_clause;
        ex_clause.constraints.extend(answer_subst.constraints);

        debug!("apply_answer_subst: ex_clause = {:?}", ex_clause);
        Ok(ex_clause)
    }
}

struct AnswerSubstitutor<'cx, 'tcx> {
    infcx: &'cx InferCtxt<'cx, 'tcx>,
    environment: Environment<'tcx>,
    answer_subst: CanonicalVarValues<'tcx>,
    binder_index: ty::DebruijnIndex,
    ex_clause: ChalkExClause<'tcx>,
}

impl AnswerSubstitutor<'cx, 'tcx> {
    fn unify_free_answer_var(
        &mut self,
        answer_var: ty::BoundVar,
        pending: Kind<'tcx>
    ) -> RelateResult<'tcx, ()> {
        let answer_param = &self.answer_subst.var_values[answer_var];
        let pending = &ty::fold::shift_out_vars(
            self.infcx.tcx,
            &pending,
            self.binder_index.as_u32()
        );

        super::into_ex_clause(
            unify(self.infcx, self.environment, ty::Variance::Invariant, answer_param, pending)?,
            &mut self.ex_clause
        );

        Ok(())
    }
}

impl TypeRelation<'tcx> for AnswerSubstitutor<'cx, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn tag(&self) -> &'static str {
        "chalk_context::answer_substitutor"
    }

    fn a_is_expected(&self) -> bool {
        true
    }

    fn relate_with_variance<T: Relate<'tcx>>(
        &mut self,
        _variance: ty::Variance,
        a: &T,
        b: &T,
    ) -> RelateResult<'tcx, T> {
        // We don't care about variance.
        self.relate(a, b)
    }

    fn binders<T: Relate<'tcx>>(
        &mut self,
        a: &ty::Binder<T>,
        b: &ty::Binder<T>,
    ) -> RelateResult<'tcx, ty::Binder<T>> {
        self.binder_index.shift_in(1);
        let result = self.relate(a.skip_binder(), b.skip_binder())?;
        self.binder_index.shift_out(1);
        Ok(ty::Binder::bind(result))
    }

    fn tys(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        let b = self.infcx.shallow_resolve(b);
        debug!("AnswerSubstitutor::tys(a = {:?}, b = {:?})", a, b);

        if let &ty::Bound(debruijn, bound_ty) = &a.sty {
            // Free bound var
            if debruijn == self.binder_index {
                self.unify_free_answer_var(bound_ty.var, b.into())?;
                return Ok(b);
            }
        }

        match (&a.sty, &b.sty) {
            (&ty::Bound(a_debruijn, a_bound), &ty::Bound(b_debruijn, b_bound)) => {
                assert_eq!(a_debruijn, b_debruijn);
                assert_eq!(a_bound.var, b_bound.var);
                Ok(a)
            }

            // Those should have been canonicalized away.
            (ty::Placeholder(..), _) => {
                bug!("unexpected placeholder ty in `AnswerSubstitutor`: {:?} ", a);
            }

            // Everything else should just be a perfect match as well,
            // and we forbid inference variables.
            _ => match ty::relate::super_relate_tys(self, a, b) {
                Ok(ty) => Ok(ty),
                Err(err) => bug!("type mismatch in `AnswerSubstitutor`: {}", err),
            }
        }
    }

    fn regions(
        &mut self,
        a: ty::Region<'tcx>,
        b: ty::Region<'tcx>,
    ) -> RelateResult<'tcx, ty::Region<'tcx>> {
        let b = match b {
            &ty::ReVar(vid) => self.infcx
                .borrow_region_constraints()
                .opportunistic_resolve_var(self.infcx.tcx, vid),

            other => other,
        };

        if let &ty::ReLateBound(debruijn, bound) = a {
            // Free bound region
            if debruijn == self.binder_index {
                self.unify_free_answer_var(bound.assert_bound_var(), b.into())?;
                return Ok(b);
            }
        }

        match (a, b) {
            (&ty::ReLateBound(a_debruijn, a_bound), &ty::ReLateBound(b_debruijn, b_bound)) => {
                assert_eq!(a_debruijn, b_debruijn);
                assert_eq!(a_bound.assert_bound_var(), b_bound.assert_bound_var());
            }

            (ty::ReStatic, ty::ReStatic) |
            (ty::ReErased, ty::ReErased) |
            (ty::ReEmpty, ty::ReEmpty) => (),

            (&ty::ReFree(a_free), &ty::ReFree(b_free)) => {
                assert_eq!(a_free, b_free);
            }

            _ => bug!("unexpected regions in `AnswerSubstitutor`: {:?}, {:?}", a, b),
        }

        Ok(a)
    }

    fn consts(
        &mut self,
        a: &'tcx ty::Const<'tcx>,
        b: &'tcx ty::Const<'tcx>,
    ) -> RelateResult<'tcx, &'tcx ty::Const<'tcx>> {
        if let ty::Const {
            val: ConstValue::Infer(InferConst::Canonical(debruijn, bound_ct)),
            ..
        } = a {
            if *debruijn == self.binder_index {
                self.unify_free_answer_var(*bound_ct, b.into())?;
                return Ok(b);
            }
        }

        match (a, b) {
            (
                ty::Const {
                    val: ConstValue::Infer(InferConst::Canonical(a_debruijn, a_bound)),
                    ..
                },
                ty::Const {
                    val: ConstValue::Infer(InferConst::Canonical(b_debruijn, b_bound)),
                    ..
                },
            ) => {
                assert_eq!(a_debruijn, b_debruijn);
                assert_eq!(a_bound, b_bound);
                Ok(a)
            }

            // Everything else should just be a perfect match as well,
            // and we forbid inference variables.
            _ => match ty::relate::super_relate_consts(self, a, b) {
                Ok(ct) => Ok(ct),
                Err(err) => bug!("const mismatch in `AnswerSubstitutor`: {}", err),
            }
        }
    }
}
