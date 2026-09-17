//! Logic for `-Zassumptions-on-binders` stuff

#[cfg(feature = "nightly")]
use rustc_data_structures::transitive_relation::TransitiveRelationBuilder;
use rustc_type_ir::inherent::*;
use rustc_type_ir::outlives::{Component, push_outlives_components};
#[cfg(not(feature = "nightly"))]
use rustc_type_ir::region_constraint::TransitiveRelationBuilder;
use rustc_type_ir::region_constraint::{
    And, Assumptions, LeafRegionConstraint, Or, RegionConstraint,
    eagerly_handle_placeholders_in_universe, propagate_ambiguity,
};
use rustc_type_ir::{
    AliasTy, Binder, ClauseKind, InferCtxtLike, Interner, Region, TypeVisitable, TypeVisitableExt,
    TypeVisitor, UniverseIndex,
};
use tracing::{debug, instrument};

use crate::delegate::SolverDelegate;
use crate::solve::{
    CanonicalResponse, Certainty, EvalCtxt, ExternalConstraintsData, ExternalRegionConstraints,
    Goal, NoSolution, QueryResultOrRerunNonErased, Response, has_only_region_constraints,
};

fn simplify_outlives_constraint<I: Interner>(
    constraint: RegionConstraint<I>,
) -> RegionConstraint<I> {
    let mut alternatives: Vec<And<I>> = Vec::new();
    for branch in constraint.or_constraint.0.iter() {
        let branch = And::new(constraint.and_constraint.0.iter().chain(branch.0.iter())
            .filter(|leaf| !matches!(leaf,
                LeafRegionConstraint::RegionOutlives(sup, sub, _) if sup == sub || sup.is_static()
            )).cloned());
        if branch.0.is_empty() {
            return RegionConstraint::new_true();
        }
        if alternatives.iter().any(|other| other.0.iter().all(|leaf| branch.0.contains(leaf))) {
            continue;
        }
        alternatives.retain(|other| !branch.0.iter().all(|leaf| other.0.contains(leaf)));
        alternatives.push(branch);
    }
    RegionConstraint::new_from_or(Or::new(alternatives))
}

/// Logic for `-Zassumptions-on-binders` stuff
impl<'a, D, I> EvalCtxt<'a, D>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    pub(in crate::solve) fn evaluate_outlives_candidate(
        &mut self,
    ) -> QueryResultOrRerunNonErased<I> {
        let response = self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)?;
        if response.value.certainty != Certainty::Yes || !has_only_region_constraints(response) {
            return Ok(response);
        }
        let certainty = self.eagerly_handle_placeholders()?;
        let mut external = ExternalConstraintsData::new(self.cx());
        external.region_constraints = ExternalRegionConstraints::NextGen(
            simplify_outlives_constraint(self.delegate.get_solver_region_constraint()),
        );
        let (var_values, external) =
            self.delegate.deeply_resolve_via_unification_table((self.var_values, external));
        Ok(crate::canonical::canonicalize_response(
            self.delegate,
            self.max_input_universe,
            Response {
                certainty,
                var_values,
                external_constraints: self.cx().mk_external_constraints(external),
            },
        ))
    }

    pub(in crate::solve) fn instantiate_outlives_binder<
        T: rustc_type_ir::TypeFoldable<I> + Copy,
    >(
        &mut self,
        binder: Binder<I, T>,
    ) -> T {
        if binder.has_bound_vars() {
            let universe = self.delegate.create_next_universe();
            self.delegate.insert_placeholder_assumptions(universe, Some(Assumptions::empty()));
        }
        self.instantiate_binder_with_infer(binder)
    }

    pub(in crate::solve) fn merge_outlives_responses(
        &mut self,
        responses: &[CanonicalResponse<I>],
    ) -> QueryResultOrRerunNonErased<I> {
        let mut alternatives = RegionConstraint::new_false();
        for response in crate::canonical::instantiate_responses_with_shared_values(
            self.delegate,
            self.var_values.var_values.as_slice(),
            responses,
            self.origin_span,
        ) {
            let mut constraint = match &response.external_constraints.region_constraints {
                ExternalRegionConstraints::NextGen(constraint) => constraint.clone(),
                ExternalRegionConstraints::Old(constraints)
                | ExternalRegionConstraints::Combined { constraints, .. } => {
                    let mut result = match &response.external_constraints.region_constraints {
                        ExternalRegionConstraints::Combined { solver_constraints, .. } => {
                            solver_constraints.clone()
                        }
                        _ => RegionConstraint::new_true(),
                    };
                    for (constraint, _) in constraints {
                        for rustc_type_ir::OutlivesClause(sup, sub) in constraint.iter_outlives() {
                            let outlives = match sup.kind() {
                                rustc_type_ir::GenericArgKind::Lifetime(sup) => {
                                    Or::new_leaf(LeafRegionConstraint::RegionOutlives(sup, sub, ()))
                                }
                                rustc_type_ir::GenericArgKind::Type(sup) => {
                                    self.destructure_type_outlives(sup, sub)
                                }
                                rustc_type_ir::GenericArgKind::Const(_) => unreachable!(),
                            };
                            result = RegionConstraint::build_and(
                                result,
                                RegionConstraint::new_from_or(outlives),
                            );
                        }
                    }
                    result
                }
            };
            for (original, result) in
                self.var_values.var_values.iter().zip(response.var_values.var_values.iter())
            {
                match (original.as_region(), result.as_region()) {
                    (Some(original), Some(result)) if original != result => {
                        constraint = RegionConstraint::build_and(
                            constraint,
                            RegionConstraint::new_from_or(Or::new([And::new([
                                LeafRegionConstraint::RegionOutlives(original, result, ()),
                                LeafRegionConstraint::RegionOutlives(result, original, ()),
                            ])])),
                        );
                    }
                    _ => debug_assert_eq!(
                        self.deeply_resolve_ignoring_regions(original),
                        self.deeply_resolve_ignoring_regions(result),
                    ),
                }
            }
            alternatives = RegionConstraint::build_or(alternatives, constraint);
        }
        self.register_solver_region_constraint(simplify_outlives_constraint(alternatives));
        self.evaluate_added_goals_and_make_canonical_response(Certainty::Yes)
    }

    /// Computes the assumptions associated with a binder for use in eagerly handling placeholders when
    /// exiting the binder. Though, right now we do not actually handle placeholders when exiting binders,
    /// instead we handle placeholders when computing the final response for the goal being computed.
    #[instrument(level = "debug", skip(self), ret)]
    pub(super) fn region_assumptions_for_placeholders_in_universe(
        &mut self,
        t: impl TypeVisitable<I>,
        u: UniverseIndex,
        param_env: I::ParamEnv,
    ) -> Option<Assumptions<I>> {
        assert!(self.cx().uses_solver_region_constraints());

        struct RawAssumptions<'a, 'b, D: SolverDelegate<Interner = I>, I: Interner> {
            ecx: &'a mut EvalCtxt<'b, D, I>,
            param_env: I::ParamEnv,
            out: Vec<Goal<I, I::Predicate>>,
        }

        impl<D, I> TypeVisitor<I> for RawAssumptions<'_, '_, D, I>
        where
            I: Interner,
            D: SolverDelegate<Interner = I>,
        {
            type Result = ();

            fn visit_ty(&mut self, t: I::Ty) {
                self.out.extend(
                    self.ecx
                        .well_formed_goals(self.param_env, t.into())
                        .unwrap_or(vec![Goal::new(
                            self.ecx.cx(),
                            self.param_env,
                            ClauseKind::WellFormed(t.into()),
                        )])
                        .into_iter(),
                );
            }

            fn visit_const(&mut self, c: I::Const) {
                self.out.extend(
                    self.ecx
                        .well_formed_goals(self.param_env, c.into())
                        .unwrap_or(vec![Goal::new(
                            self.ecx.cx(),
                            self.param_env,
                            ClauseKind::WellFormed(c.into()),
                        )])
                        .into_iter(),
                );
            }
        }

        let mut reqs_builder = RawAssumptions { ecx: self, param_env, out: vec![] };
        t.visit_with(&mut reqs_builder);
        let reqs = reqs_builder.out;

        // If there are inference variables in type outlives then we may not be able
        // to elaborate to the full set of implied bounds right now. To avoid incorrectly
        // NoSolution'ing when lifting constraints to a lower universe due to no usable
        // assumptions, we just bail here.
        //
        // This is somewhat imprecise as if both the infer var and the outlived region are
        // in a lower universe than the binder we're computing assumptions for then it doesn't
        // really matter as we wouldn't use those outlives as assumptions anyway.
        if reqs.iter().any(|goal| {
            // We don't care about region infers as they can't be further destructured
            goal.predicate.has_non_region_infer()
        }) {
            return None;
        }

        // FIXME(-Zassumptions-on-binders): we need to normalize here/somewhere
        // as we assume the type outlives assumptions only have rigid types :>
        //
        // `Assumptions::new` elaborates, restricts the clauses to `u` and picks out the
        // outlives ones for us, so we just hand over everything the requirements gave us.
        let clauses = reqs.into_iter().filter_map(|goal| goal.predicate.as_clause());

        Some(Assumptions::new(
            &**self.delegate,
            clauses,
            TransitiveRelationBuilder::default().freeze(),
            u,
        ))
    }

    #[instrument(level = "debug", skip(self), ret)]
    pub(super) fn eagerly_handle_placeholders(&mut self) -> Result<Certainty, NoSolution> {
        let constraint = self.delegate.get_solver_region_constraint();
        // Structural type relations still record region equalities in the
        // ordinary collector. They must participate in binder checking and
        // in the response just like explicitly registered outlives goals.
        let relations = self.delegate.make_deduplicated_region_constraints();
        let relations = relations
            .into_iter()
            .filter(|(constraint, _)| !constraint.is_trivial())
            .flat_map(|(constraint, _)| {
                constraint.iter_outlives().map(|rustc_type_ir::OutlivesClause(sup, sub)| match sup
                    .kind()
                {
                    rustc_type_ir::GenericArgKind::Lifetime(sup) => {
                        LeafRegionConstraint::RegionOutlives(sup, sub, ())
                    }
                    rustc_type_ir::GenericArgKind::Type(sup) => {
                        LeafRegionConstraint::PlaceholderTyOutlives(sup, sub, ())
                    }
                    rustc_type_ir::GenericArgKind::Const(_) => unreachable!(),
                })
            });
        let constraint = RegionConstraint::build_and(
            constraint,
            RegionConstraint { and_constraint: And::new(relations), or_constraint: Or::new_true() },
        );

        let smallest_universe = self.max_input_universe.index();
        let largest_universe = self.delegate.universe().index();
        debug!(?smallest_universe, largest_universe);

        let constraint = ((smallest_universe + 1)..=largest_universe)
            .map(|u| UniverseIndex::from_usize(u))
            .rev()
            .fold(constraint, |constraint, u| {
                eagerly_handle_placeholders_in_universe(&**self.delegate, constraint, u)
            });
        let constraint = propagate_ambiguity(constraint);

        debug!("final constraint={:?}", constraint);
        self.delegate.overwrite_solver_region_constraint(constraint.clone(), self.origin_span);

        if constraint.is_false() {
            Err(NoSolution)
        } else if constraint.is_ambig() {
            Ok(Certainty::AMBIGUOUS)
        } else {
            Ok(Certainty::Yes)
        }
    }

    /// Convert a type outlives constraint into a set of region outlives constraints and
    /// type outlives constraints between the "components" of the type. E.g. `Foo<T, 'a>: 'b`
    /// will be turned into `T: 'b, 'a: 'b`
    #[instrument(level = "debug", skip(self), ret)]
    pub(in crate::solve) fn destructure_type_outlives(&mut self, ty: I::Ty, r: Region<I>) -> Or<I> {
        let mut components = Default::default();
        push_outlives_components(self.cx(), ty, &mut components);
        self.destructure_components(&components, r)
    }

    fn destructure_components(&mut self, components: &[Component<I>], r: Region<I>) -> Or<I> {
        components
            .into_iter()
            .fold(Or::new_true(), |acc, c| Or::build_and(acc, self.destructure_component(c, r)))
    }

    fn destructure_component(&mut self, c: &Component<I>, r: Region<I>) -> Or<I> {
        use Component::*;
        use LeafRegionConstraint::*;
        match c {
            Region(c_r) => Or::new_leaf(RegionOutlives(*c_r, r, ())),
            Placeholder(p) => {
                Or::new_leaf(PlaceholderTyOutlives(Ty::new_placeholder(self.cx(), *p), r, ()))
            }
            Alias(_, alias) => self.destructure_alias_outlives(*alias, r),
            UnresolvedInferenceVariable(_) => Or::new_ambig(()),
            Param(_) => panic!("Params should have been canonicalized to placeholders"),
            EscapingAlias(components) => self.destructure_components(components, r),
        }
    }

    /// Convert an alias outlives constraint into an OR constraint of any number of three
    /// separate classes of candidates:
    /// 1. component outlives. we turn `Alias<T, 'a>: 'b` into `T: 'b, 'a: 'b`.
    /// 2. item bounds. we turn `Alias<T, 'a>: 'b` into `'c: 'b` if `Alias` is
    ///     defined as `type Alias<T, 'a>: 'c`
    /// 3. env assumptions. we defer handling `Alias<T, 'a>: 'b` via where clauses until
    ///     when exiting the current binder. See [`LeafRegionConstraint::AliasTyOutlivesViaEnv`].
    #[instrument(level = "debug", skip(self), ret)]
    fn destructure_alias_outlives(&mut self, alias: AliasTy<I>, r: Region<I>) -> Or<I> {
        use LeafRegionConstraint::*;

        let item_bounds =
            rustc_type_ir::outlives::declared_bounds_from_definition(self.cx(), alias)
                .map(|bound| And::new([RegionOutlives(bound, r, ())]));
        let item_bound_outlives = Or::new(item_bounds);

        let where_clause_outlives =
            Or::new_leaf(AliasTyOutlivesViaEnv(Binder::dummy((alias, r)), ()));

        let mut components = Default::default();
        rustc_type_ir::outlives::compute_alias_components_recursive(
            self.cx(),
            alias,
            &mut components,
        );
        let components_outlives = self.destructure_components(&components, r);

        let assumption_outlives = Or::build_or(item_bound_outlives, where_clause_outlives);
        Or::build_or(assumption_outlives, components_outlives)
    }
}
