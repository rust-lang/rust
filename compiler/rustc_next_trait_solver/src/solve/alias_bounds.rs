//! Use declaration bounds after an equality replaces a projection.

use rustc_type_ir::data_structures::HashSet;
use rustc_type_ir::inherent::*;
use rustc_type_ir::outlives::{
    Component, compute_alias_components_recursive, push_outlives_components,
};
use rustc_type_ir::search_graph::LowerAvailableDepth;
use rustc_type_ir::solve::{AliasBoundKind, RerunNonErased, RerunResultExt};
use rustc_type_ir::{
    self as ty, Interner, TypeFoldable, TypeFolder, TypeSuperFoldable, TypeVisitableExt,
    Unnormalized, Upcast,
};

use super::assembly::{Candidate, GoalKind};
use super::{
    CandidateSource, Certainty, EvalCtxt, Goal, GoalSource, NestedNormalizationGoals, NoSolution,
    ParamEnvSource, QueryResultOrRerunNonErased, has_no_inference_or_external_constraints,
    has_only_region_constraints,
};
use crate::delegate::SolverDelegate;

impl<D: SolverDelegate<Interner = I>, I: Interner> EvalCtxt<'_, D> {
    pub(super) fn compute_type_outlives_goal(
        &mut self,
        goal: Goal<I, ty::OutlivesClause<I, I::Ty>>,
    ) -> QueryResultOrRerunNonErased<I> {
        let cx = self.cx();
        if !cx.next_trait_solver_globally() || goal.predicate.0.has_non_region_infer() {
            return self.compute_type_outlives_goal_structurally(goal);
        }
        let sources: Vec<_> = goal
            .param_env
            .caller_bounds()
            .filter_map(|clause| clause.as_projection_clause())
            .filter(|&projection| {
                declaration_reaches_goal(cx, projection.upcast(cx), goal.predicate.upcast(cx))
            })
            .collect();
        if sources.is_empty() {
            return self.compute_type_outlives_goal_structurally(goal);
        }
        let fallback = self
            .probe_trait_candidate(CandidateSource::AliasBound(AliasBoundKind::SelfBounds))
            .enter(|ecx| ecx.compute_type_outlives_goal_structurally(goal))
            .map_err_to_rerun()?;
        let mut candidates = Vec::new();
        if let Ok(candidate) = fallback {
            if !has_only_region_constraints(candidate.result)
                || (candidate.result.value.certainty == Certainty::Yes
                    && has_no_inference_or_external_constraints(candidate.result))
            {
                return Ok(candidate.result);
            }
            candidates.push(candidate);
        }
        let components = self
            .probe_trait_candidate(CandidateSource::AliasBound(AliasBoundKind::SelfBounds))
            .enter(|ecx| ecx.prove_outlives_components(goal))
            .map_err_to_rerun()?;
        if let Ok(candidate) = components {
            if candidate.result.value.certainty == Certainty::Yes
                && has_no_inference_or_external_constraints(candidate.result)
            {
                return Ok(candidate.result);
            }
            candidates.push(candidate);
        }
        self.projection_declaration_candidates(
            goal.with(cx, goal.predicate),
            sources,
            &mut candidates,
        )?;
        self.merge_outlives_candidates(candidates)
    }

    pub(super) fn assemble_declaration_candidates<G: GoalKind<D>>(
        &mut self,
        goal: Goal<I, G>,
        candidates: &mut Vec<Candidate<I>>,
    ) -> Result<(), RerunNonErased> {
        let cx = self.cx();
        if !cx.next_trait_solver_globally() {
            return Ok(());
        }
        // These are consequences of existing bounds. They do not override an
        // applicable direct bound or a proof which adds no constraints.
        if candidates.iter().any(|candidate| {
            (has_no_inference_or_external_constraints(candidate.result)
                || matches!(
                    candidate.source,
                    CandidateSource::ParamEnv(ParamEnvSource::NonGlobal)
                        | CandidateSource::AliasBound(_)
                ))
                && candidate.result.value.certainty == Certainty::Yes
        }) {
            return Ok(());
        }
        let goal = goal.with(cx, goal.predicate.as_predicate(cx));
        let sources = goal
            .param_env
            .caller_bounds()
            .filter_map(|clause| clause.as_projection_clause())
            .filter(|projection| projection.skip_binder().term.as_type().is_some())
            .filter(|projection| {
                cx.explicit_item_self_bounds(projection.skip_binder().def_id().into())
                    .iter_identity()
                    .map(Unnormalized::skip_norm_wip)
                    .any(|bound| declaration_reaches_goal(cx, bound, goal.predicate))
            })
            .collect();
        let mut declared = Vec::new();
        self.projection_declaration_candidates(goal, sources, &mut declared)?;
        if let Some(first) = declared.first()
            && let Ok(mut candidate) = self
                .probe_trait_candidate(first.source)
                .enter(|ecx| ecx.merge_equivalent_declarations(&declared)?.ok_or(NoSolution.into()))
                .map_err_to_rerun()?
        {
            for declared in declared {
                candidate.head_usages.merge_usages(declared.head_usages);
            }
            candidates.push(candidate);
        } else {
            candidates.extend(declared);
        }
        Ok(())
    }

    fn projection_declaration_candidates(
        &mut self,
        goal: Goal<I, I::Predicate>,
        sources: Vec<ty::Binder<I, ty::ProjectionClause<I>>>,
        candidates: &mut Vec<Candidate<I>>,
    ) -> Result<(), RerunNonErased> {
        let cx = self.cx();
        for projection in sources {
            let value = projection.skip_binder();
            let check_self = value.projection_term.self_ty().has_escaping_bound_vars();
            if cx.is_impl_trait_in_trait(value.def_id().into())
                && let Some(output) = value.term.as_type()
                && ty::set_aliases_to_non_rigid(cx, output).skip_norm_wip()
                    == cx
                        .type_of(value.def_id().into())
                        .instantiate(cx, value.projection_term.args)
                        .skip_norm_wip()
            {
                // This links a method's associated return type to the opaque
                // whose definition is being checked. It is a normalization
                // equation, not independent evidence for that definition's bounds.
                continue;
            }
            // Impl checking installs equalities for normalization before their
            // declaration bounds have been proved. Require an independent trait
            // assumption before using an equality as declaration evidence.
            for assumption in goal.param_env.caller_bounds().filter(|c| {
                c.as_trait_clause().is_some_and(|clause| {
                    clause.polarity() == ty::ClausePolarity::Positive
                        && clause.def_id()
                            == projection.skip_binder().projection_term.trait_ref(cx).def_id
                })
            }) {
                let source = if goal.predicate.is_global() {
                    CandidateSource::ParamEnv(ParamEnvSource::Global)
                } else {
                    CandidateSource::AliasBound(AliasBoundKind::NonSelfBounds)
                };
                let candidate = self
                    .probe_trait_candidate(source)
                    .enter(|ecx| {
                        let projection = ecx.instantiate_outlives_binder(projection);
                        ty::TraitClause::match_assumption(
                            ecx,
                            goal.with(cx, projection.projection_term.trait_ref(cx)),
                            assumption,
                            |ecx| {
                                if check_self {
                                    let goals = ecx
                                        .well_formed_goals(
                                            goal.param_env,
                                            projection.projection_term.self_ty().into(),
                                        )
                                        .ok_or(NoSolution)?;
                                    ecx.add_goals(GoalSource::AliasWellFormed, goals)?;
                                }
                                let certainty = ecx.add_bound_from_clause(
                                    goal,
                                    ty::ClauseKind::Projection(projection),
                                )?;
                                ecx.finish_declaration_candidate(goal, certainty)
                            },
                        )
                    })
                    .map_err_to_rerun()?;
                if let Ok(candidate) = candidate {
                    candidates.push(candidate);
                }
            }
        }
        Ok(())
    }

    fn prove_outlives_components(
        &mut self,
        goal: Goal<I, ty::OutlivesClause<I, I::Ty>>,
    ) -> QueryResultOrRerunNonErased<I> {
        let cx = self.cx();
        let ty = self.normalize(
            GoalSource::Misc,
            goal.param_env,
            Unnormalized::new_wip(goal.predicate.0),
        )?;
        let certainty = self.try_evaluate_added_goals()?;
        let ty = self.deeply_resolve_ignoring_regions(ty);
        if certainty != Certainty::Yes || ty.has_non_region_infer() || ty.has_non_rigid_aliases() {
            return self.evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS);
        }
        if matches!(ty.kind(), ty::Param(_) | ty::Placeholder(_)) {
            return Err(NoSolution.into());
        }
        let mut components = Default::default();
        if let ty::Alias(_, alias) = ty.kind() {
            compute_alias_components_recursive(cx, alias, &mut components);
        } else {
            push_outlives_components(cx, ty, &mut components);
        }
        let mut components = components.into_vec();
        while let Some(component) = components.pop() {
            let ty = match component {
                Component::Region(region) => {
                    self.add_goal(
                        GoalSource::Misc,
                        goal.with(cx, ty::OutlivesClause(region, goal.predicate.1)),
                    )?;
                    continue;
                }
                Component::Param(param) => Ty::new_param(cx, param),
                Component::Placeholder(placeholder) => Ty::new_placeholder(cx, placeholder),
                Component::Alias(is_rigid, alias) => alias.to_ty(cx, is_rigid),
                Component::EscapingAlias(nested) => {
                    components.extend(nested);
                    continue;
                }
                Component::UnresolvedInferenceVariable(_) => {
                    return self
                        .evaluate_added_goals_and_make_canonical_response(Certainty::AMBIGUOUS);
                }
            };
            self.add_goal(
                GoalSource::TypeRelating,
                goal.with(cx, ty::OutlivesClause(ty, goal.predicate.1)),
            )?;
        }
        self.evaluate_outlives_candidate()
    }

    /// The caller supplies one established clause. Its consequences remain
    /// local to this proof, including each supertrait path's WF requirements.
    pub(super) fn compute_bound_from_clause(
        &mut self,
        goal: Goal<I, I::Predicate>,
        source: I::Clause,
    ) -> QueryResultOrRerunNonErased<I> {
        let cx = self.cx();
        let source = source.kind().no_bound_vars().unwrap();
        let args = match source {
            ty::ClauseKind::Trait(clause) => clause.trait_ref.args,
            ty::ClauseKind::Projection(clause) => clause.projection_term.args,
            ty::ClauseKind::TypeOutlives(_) => {
                return self.match_declaration_clause(goal, source);
            }
            _ => return Err(NoSolution.into()),
        };
        // The established source already supplies Self's WF. Requiring it again
        // can depend on the declaration consequence we are proving. A quantified
        // source's Self is checked when its binder is instantiated instead.
        for term in args.iter().skip(1).filter_map(|arg| arg.as_term()) {
            let goals = self.well_formed_goals(goal.param_env, term).ok_or(NoSolution)?;
            self.add_goals(GoalSource::AliasWellFormed, goals)?;
        }
        let (bounds, replacement): (Vec<_>, _) = match source {
            ty::ClauseKind::Trait(clause) => {
                if clause.polarity != ty::ClausePolarity::Positive {
                    return Err(NoSolution.into());
                }
                let bounds = cx
                    .explicit_super_clauses_of(clause.def_id())
                    .iter_instantiated(cx, args)
                    .map(|clause| clause.skip_norm_wip().0)
                    .collect();
                (bounds, None)
            }
            ty::ClauseKind::Projection(clause) => {
                let ty::AliasTermKind::ProjectionTy { def_id } = clause.projection_term.kind else {
                    return Err(NoSolution.into());
                };
                let Some(value) = clause.term.as_type() else { return Err(NoSolution.into()) };
                // The established equality supplies the output type. Checking
                // its WF here would demand the same declaration consequences
                // that this proof is establishing.
                self.add_goals(
                    GoalSource::AliasWellFormed,
                    cx.own_clauses_of(def_id.into())
                        .iter_instantiated(cx, args)
                        .map(|clause| goal.with(cx, clause.skip_norm_wip())),
                )?;
                let bounds = cx
                    .explicit_item_self_bounds(def_id.into())
                    .iter_instantiated(cx, args)
                    .map(Unnormalized::skip_norm_wip)
                    .collect();
                (bounds, Some((clause.projection_term.expect_ty(), value)))
            }
            _ => unreachable!(),
        };
        let direct = if clause_matches_goal(source, goal.predicate) {
            self.probe_trait_candidate(CandidateSource::AliasBound(AliasBoundKind::SelfBounds))
                .enter(|ecx| ecx.match_declaration_clause(goal, source))
                .map_err_to_rerun()?
                .ok()
        } else {
            None
        };
        if let Some(candidate) = &direct
            && candidate.result.value.certainty == Certainty::Yes
            && has_no_inference_or_external_constraints(candidate.result)
        {
            return Ok(candidate.result);
        }
        let mut candidates: Vec<_> = direct.into_iter().collect();
        let mut bounds = bounds;
        bounds.sort_by_key(|bound| bound.as_type_outlives_clause().is_none());
        for bound in bounds {
            if !declaration_reaches_goal(cx, bound, goal.predicate) {
                continue;
            }
            let candidate = self
                .probe_trait_candidate(CandidateSource::AliasBound(AliasBoundKind::SelfBounds))
                .enter(|ecx| {
                    let mut clause = ecx.instantiate_outlives_binder(bound.kind());
                    if let Some((alias, value)) = replacement {
                        clause = clause.fold_with(&mut ReplaceProjection { cx, alias, value });
                    }
                    let certainty = ecx.add_bound_from_clause(goal, clause)?;
                    ecx.finish_declaration_candidate(goal, certainty)
                })
                .map_err_to_rerun()?;
            if let Ok(candidate) = candidate {
                if candidate.result.value.certainty == Certainty::Yes
                    && has_no_inference_or_external_constraints(candidate.result)
                {
                    return Ok(candidate.result);
                }
                candidates.push(candidate);
            }
        }
        self.merge_declaration_candidates(goal, candidates)
    }

    fn add_bound_from_clause(
        &mut self,
        goal: Goal<I, I::Predicate>,
        clause: ty::ClauseKind<I>,
    ) -> Result<Certainty, ty::solve::NoSolutionOrRerunNonErased> {
        let cx = self.cx();
        let source_args = match clause {
            ty::ClauseKind::Projection(clause) => Some(clause.projection_term.args),
            ty::ClauseKind::Trait(clause) => Some(clause.trait_ref.args),
            _ => None,
        };
        if let ty::PredicateKind::NormalizesTo(target) = goal.predicate.kind().skip_binder()
            && let Some(args) = source_args
            && args.iter().flat_map(ty::walk::TypeWalker::<I>::new).any(|arg| {
                arg.as_type().is_some_and(|ty| {
                    matches!(ty.kind(), ty::Alias(_, alias) if ty::AliasTerm::from(alias) == target.alias)
                })
            })
        {
            // Matching this declaration requires the very projection whose
            // value is being computed. It cannot supply an independent
            // normalization candidate for that value.
            return Err(NoSolution.into());
        }
        let clause: I::Clause = if let ty::ClauseKind::Projection(mut projection) = clause {
            projection.projection_term.args = self.normalize(
                GoalSource::TypeRelating,
                goal.param_env,
                Unnormalized::new_wip(projection.projection_term.args),
            )?;
            ty::Binder::dummy(projection).upcast(cx)
        } else {
            self.normalize(
                GoalSource::TypeRelating,
                goal.param_env,
                Unnormalized::new_wip(ty::Binder::dummy(clause).upcast(cx)),
            )?
        };
        let (NestedNormalizationGoals(nested), result) = self.evaluate_goal_raw(
            GoalSource::TypeRelating,
            goal.with(cx, ty::PredicateKind::BoundFromClause(clause, goal.predicate)),
            LowerAvailableDepth::Yes,
        )?;
        for (source, goal) in nested {
            self.add_goal(source, goal)?;
        }
        Ok(result.certainty)
    }

    fn match_declaration_clause(
        &mut self,
        goal: Goal<I, I::Predicate>,
        clause: ty::ClauseKind<I>,
    ) -> QueryResultOrRerunNonErased<I> {
        let cx = self.cx();
        match (clause, goal.predicate.kind().skip_binder()) {
            (
                ty::ClauseKind::Trait(source),
                ty::PredicateKind::Clause(ty::ClauseKind::Trait(target)),
            ) => {
                self.eq(goal.param_env, source.trait_ref, target.trait_ref)?;
                self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            }
            (ty::ClauseKind::Projection(mut source), ty::PredicateKind::NormalizesTo(target)) => {
                self.eq(goal.param_env, source.projection_term, target.alias)?;
                // Preserve the direction of explicit equalities. In particular,
                // an inferred reverse of `A = B` reduces to `B = B`, which does
                // not provide a normalization candidate for `B`.
                source.term = source.term.fold_with(&mut EnvironmentValues {
                    cx,
                    param_env: goal.param_env,
                    active: HashSet::default(),
                });
                if source.term.as_type().is_some_and(|ty| {
                    matches!(ty.kind(), ty::Alias(_, alias) if ty::AliasTerm::from(alias) == target.alias)
                }) { return Err(NoSolution.into()); }
                let term = self.normalize(
                    GoalSource::Misc,
                    goal.param_env,
                    Unnormalized::new_wip(source.term),
                )?;
                self.eq(goal.param_env, term, target.term)?;
                self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
            }
            (
                ty::ClauseKind::TypeOutlives(ty::OutlivesClause(subject, region)),
                ty::PredicateKind::Clause(ty::ClauseKind::TypeOutlives(target)),
            ) => {
                self.eq(goal.param_env, subject, target.0)?;
                self.add_goal(
                    GoalSource::Misc,
                    goal.with(cx, ty::OutlivesClause(region, target.1)),
                )?;
                self.evaluate_outlives_candidate()
            }
            _ => Err(NoSolution.into()),
        }
    }

    fn finish_declaration_candidate(
        &mut self,
        goal: Goal<I, I::Predicate>,
        certainty: Certainty,
    ) -> QueryResultOrRerunNonErased<I> {
        if certainty == Certainty::Yes
            && matches!(
                goal.predicate.kind().skip_binder(),
                ty::PredicateKind::Clause(ty::ClauseKind::TypeOutlives(_))
            )
        {
            self.evaluate_outlives_candidate()
        } else {
            self.evaluate_added_goals_and_make_canonical_response(certainty)
        }
    }

    fn merge_declaration_candidates(
        &mut self,
        goal: Goal<I, I::Predicate>,
        candidates: Vec<Candidate<I>>,
    ) -> QueryResultOrRerunNonErased<I> {
        if matches!(
            goal.predicate.kind().skip_binder(),
            ty::PredicateKind::Clause(ty::ClauseKind::TypeOutlives(_))
        ) {
            return self.merge_outlives_candidates(candidates);
        }
        if let Some((response, _)) = self.try_merge_candidates(&candidates) {
            Ok(response)
        } else if let Some(response) = self.merge_equivalent_declarations(&candidates)? {
            Ok(response)
        } else {
            self.flounder(&candidates).map_err(Into::into)
        }
    }

    fn merge_equivalent_declarations(
        &mut self,
        candidates: &[Candidate<I>],
    ) -> Result<Option<super::CanonicalResponse<I>>, ty::solve::NoSolutionOrRerunNonErased> {
        let Some(first) = candidates.first() else { return Ok(None) };
        let first = first.result;
        if candidates.len() < 2
            || !candidates.iter().all(|candidate| {
                let response = candidate.result;
                response.value.certainty == Certainty::Yes
                    && response.var_kinds == first.var_kinds
                    && response.max_universe == first.max_universe
                    && response.value.var_values == first.value.var_values
                    && response.value.external_constraints.opaque_types.is_empty()
                    && response.value.external_constraints.normalization_nested_goals.is_empty()
            })
        {
            return Ok(None);
        }
        let responses: Vec<_> = candidates.iter().map(|candidate| candidate.result).collect();
        self.merge_outlives_responses(&responses).map(Some)
    }

    fn merge_outlives_candidates(
        &mut self,
        candidates: Vec<Candidate<I>>,
    ) -> QueryResultOrRerunNonErased<I> {
        let responses: Vec<_> = candidates
            .iter()
            .filter_map(|candidate| {
                (candidate.result.value.certainty == Certainty::Yes
                    && has_only_region_constraints(candidate.result))
                .then_some(candidate.result)
            })
            .collect();
        match responses.as_slice() {
            [] => {
                if let Some((response, _)) = self.try_merge_candidates(&candidates) {
                    Ok(response)
                } else {
                    self.flounder(&candidates).map_err(Into::into)
                }
            }
            [response] => Ok(*response),
            _ => self.merge_outlives_responses(&responses),
        }
    }
}

fn declaration_reaches_goal<I: Interner>(cx: I, source: I::Clause, goal: I::Predicate) -> bool {
    let mut pending = vec![source];
    let mut traits = HashSet::default();
    let mut projections = HashSet::default();
    while let Some(clause) = pending.pop() {
        if clause_matches_goal(clause.kind().skip_binder(), goal) {
            return true;
        }
        match clause.kind().skip_binder() {
            ty::ClauseKind::Trait(clause)
                if clause.polarity == ty::ClausePolarity::Positive
                    && traits.insert(clause.def_id()) =>
            {
                pending.extend(
                    cx.explicit_super_clauses_of(clause.def_id())
                        .iter_identity()
                        .map(|clause| clause.skip_norm_wip().0),
                );
            }
            ty::ClauseKind::Projection(clause)
                if clause.term.as_type().is_some() && projections.insert(clause.def_id()) =>
            {
                pending.extend(
                    cx.explicit_item_self_bounds(clause.def_id().into())
                        .iter_identity()
                        .map(Unnormalized::skip_norm_wip),
                );
            }
            _ => {}
        }
    }
    false
}

struct ReplaceProjection<I: Interner> {
    cx: I,
    alias: ty::AliasTy<I>,
    value: I::Ty,
}

impl<I: Interner> TypeFolder<I> for ReplaceProjection<I> {
    fn cx(&self) -> I {
        self.cx
    }

    fn fold_ty(&mut self, ty: I::Ty) -> I::Ty {
        if let ty::Alias(_, alias) = ty.kind()
            && alias == self.alias
        {
            self.value
        } else {
            ty.super_fold_with(self)
        }
    }
}

fn clause_matches_goal<I: Interner>(clause: ty::ClauseKind<I>, goal: I::Predicate) -> bool {
    match (clause, goal.kind().skip_binder()) {
        (
            ty::ClauseKind::Trait(source),
            ty::PredicateKind::Clause(ty::ClauseKind::Trait(target)),
        ) => source.def_id() == target.def_id() && source.polarity == target.polarity,
        (ty::ClauseKind::Projection(source), ty::PredicateKind::NormalizesTo(target)) => {
            source.projection_term.kind == target.alias.kind
        }
        (
            ty::ClauseKind::TypeOutlives(_),
            ty::PredicateKind::Clause(ty::ClauseKind::TypeOutlives(_)),
        ) => true,
        _ => false,
    }
}

/// Substitute exact environment equations without invoking declaration search.
/// Quantified equations are left to the ordinary candidate machinery.
struct EnvironmentValues<I: Interner> {
    cx: I,
    param_env: I::ParamEnv,
    active: HashSet<ty::AliasTy<I>>,
}

impl<I: Interner> TypeFolder<I> for EnvironmentValues<I> {
    fn cx(&self) -> I {
        self.cx
    }
    fn fold_ty(&mut self, ty: I::Ty) -> I::Ty {
        let ty = ty.super_fold_with(self);
        let ty::Alias(_, alias) = ty.kind() else {
            return ty;
        };
        if !self.active.insert(alias) {
            return ty;
        }
        let replacement = self
            .param_env
            .caller_bounds()
            .filter_map(|c| c.as_projection_clause()?.no_bound_vars())
            .find(|c| c.projection_term == alias.into())
            .and_then(|c| c.term.as_type());
        let value = replacement.map_or(ty, |value| value.fold_with(self));
        self.active.remove(&alias);
        value
    }
}
