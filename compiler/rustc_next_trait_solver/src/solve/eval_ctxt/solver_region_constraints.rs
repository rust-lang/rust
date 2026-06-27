//! Logic for `-Zassumptions-on-binders` stuff

#[cfg(feature = "nightly")]
use rustc_data_structures::transitive_relation::TransitiveRelationBuilder;
use rustc_type_ir::ClauseKind::*;
use rustc_type_ir::inherent::*;
use rustc_type_ir::outlives::{Component, push_outlives_components};
#[cfg(not(feature = "nightly"))]
use rustc_type_ir::region_constraint::TransitiveRelationBuilder;
use rustc_type_ir::region_constraint::{
    Assumptions, RegionConstraint, eagerly_handle_placeholders_in_universe, max_universe,
};
use rustc_type_ir::{
    AliasTy, Binder, ClauseKind, InferCtxtLike, Interner, OutlivesPredicate, TypeVisitable,
    TypeVisitableExt, TypeVisitor, UniverseIndex,
};
use tracing::{debug, instrument};

use crate::delegate::SolverDelegate;
use crate::solve::{Certainty, EvalCtxt, Goal, NoSolution};

/// Logic for `-Zassumptions-on-binders` stuff
impl<'a, D, I> EvalCtxt<'a, D>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
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
        assert!(self.cx().assumptions_on_binders());

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

        let mut region_outlives_builder = TransitiveRelationBuilder::default();
        let mut type_outlives = vec![];

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
        let clauses = rustc_type_ir::elaborate::elaborate(
            self.cx(),
            reqs.into_iter().filter_map(|goal| goal.predicate.as_clause()),
        );

        clauses.filter(move |clause| max_universe(&**self.delegate, *clause) == u).for_each(
            |clause| match clause.kind().skip_binder() {
                RegionOutlives(OutlivesPredicate(r1, r2)) => {
                    assert!(clause.kind().no_bound_vars().is_some());
                    region_outlives_builder.add(r1, r2);
                }
                TypeOutlives(p) => {
                    type_outlives.push(clause.kind().map_bound(|_| p));
                }
                _ => (),
            },
        );

        Some(Assumptions::new(type_outlives, region_outlives_builder.freeze()))
    }

    #[instrument(level = "debug", skip(self), ret)]
    pub(super) fn eagerly_handle_placeholders(&mut self) -> Result<Certainty, NoSolution> {
        let constraint = self.delegate.get_solver_region_constraint();

        let smallest_universe = self.max_input_universe.index();
        let largest_universe = self.delegate.universe().index();
        debug!(?smallest_universe, largest_universe);

        let constraint = ((smallest_universe + 1)..=largest_universe)
            .map(|u| UniverseIndex::from_usize(u))
            .rev()
            .fold(constraint, |constraint, u| {
                eagerly_handle_placeholders_in_universe(&**self.delegate, constraint, u)
            });

        self.delegate.overwrite_solver_region_constraint(constraint.clone());

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
    pub(in crate::solve) fn destructure_type_outlives(
        &mut self,
        ty: I::Ty,
        r: I::Region,
    ) -> RegionConstraint<I> {
        let mut components = Default::default();
        push_outlives_components(self.cx(), ty, &mut components);
        self.destructure_components(&components, r)
    }

    fn destructure_components(
        &mut self,
        components: &[Component<I>],
        r: I::Region,
    ) -> RegionConstraint<I> {
        RegionConstraint::And(
            components.into_iter().map(|c| self.destructure_component(c, r)).collect(),
        )
    }

    fn destructure_component(&mut self, c: &Component<I>, r: I::Region) -> RegionConstraint<I> {
        use Component::*;
        match c {
            Region(c_r) => RegionConstraint::RegionOutlives(*c_r, r),
            Placeholder(p) => {
                RegionConstraint::PlaceholderTyOutlives(Ty::new_placeholder(self.cx(), *p), r)
            }
            Alias(alias) => self.destructure_alias_outlives(*alias, r),
            UnresolvedInferenceVariable(_) => RegionConstraint::Ambiguity,
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
    ///     when exiting the current binder. See [`RegionConstraint::AliasTyOutlivesViaEnv`].
    #[instrument(level = "debug", skip(self), ret)]
    fn destructure_alias_outlives(
        &mut self,
        alias: AliasTy<I>,
        r: I::Region,
    ) -> RegionConstraint<I> {
        let item_bounds =
            rustc_type_ir::outlives::declared_bounds_from_definition(self.cx(), alias)
                .map(|bound| RegionConstraint::RegionOutlives(bound, r));
        let item_bound_outlives = RegionConstraint::Or(item_bounds.collect());

        let where_clause_outlives =
            RegionConstraint::AliasTyOutlivesViaEnv(Binder::dummy((alias, r)));

        let mut components = Default::default();
        rustc_type_ir::outlives::compute_alias_components_recursive(
            self.cx(),
            alias,
            &mut components,
        );
        let components_outlives = self.destructure_components(&components, r);

        RegionConstraint::Or(Box::new([
            item_bound_outlives,
            where_clause_outlives,
            components_outlives,
        ]))
    }
}
