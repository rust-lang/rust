/// Canonicalization is used to separate some goal from its context,
/// throwing away unnecessary information in the process.
///
/// This is necessary to cache goals containing inference variables
/// and placeholders without restricting them to the current `InferCtxt`.
///
/// Canonicalization is fairly involved, for more details see the relevant
/// section of the [rustc-dev-guide][c].
///
/// [c]: https://rustc-dev-guide.rust-lang.org/solve/canonicalization.html
use super::{CanonicalGoal, Certainty, EvalCtxt, Goal};
use crate::solve::canonicalize::{CanonicalizeMode, Canonicalizer};
use crate::solve::{CanonicalResponse, QueryResult, Response};
use rustc_infer::infer::canonical::query_response::make_query_region_constraints;
use rustc_infer::infer::canonical::CanonicalVarValues;
use rustc_infer::infer::canonical::{CanonicalExt, QueryRegionConstraints};
use rustc_middle::traits::query::NoSolution;
use rustc_middle::traits::solve::{ExternalConstraints, ExternalConstraintsData};
use rustc_middle::ty::{self, GenericArgKind};
use rustc_span::DUMMY_SP;
use std::iter;
use std::ops::Deref;

impl<'tcx> EvalCtxt<'_, 'tcx> {
    /// Canonicalizes the goal remembering the original values
    /// for each bound variable.
    pub(super) fn canonicalize_goal(
        &self,
        goal: Goal<'tcx, ty::Predicate<'tcx>>,
    ) -> (Vec<ty::GenericArg<'tcx>>, CanonicalGoal<'tcx>) {
        let mut orig_values = Default::default();
        let canonical_goal = Canonicalizer::canonicalize(
            self.infcx,
            CanonicalizeMode::Input,
            &mut orig_values,
            goal,
        );
        (orig_values, canonical_goal)
    }

    /// To return the constraints of a canonical query to the caller, we canonicalize:
    ///
    /// - `var_values`: a map from bound variables in the canonical goal to
    ///   the values inferred while solving the instantiated goal.
    /// - `external_constraints`: additional constraints which aren't expressable
    ///   using simple unification of inference variables.
    #[instrument(level = "debug", skip(self))]
    pub(in crate::solve) fn evaluate_added_goals_and_make_canonical_response(
        &mut self,
        certainty: Certainty,
    ) -> QueryResult<'tcx> {
        let goals_certainty = self.try_evaluate_added_goals()?;
        let certainty = certainty.unify_and(goals_certainty);

        let external_constraints = self.compute_external_query_constraints()?;

        let response = Response { var_values: self.var_values, external_constraints, certainty };
        let canonical = Canonicalizer::canonicalize(
            self.infcx,
            CanonicalizeMode::Response { max_input_universe: self.max_input_universe },
            &mut Default::default(),
            response,
        );
        Ok(canonical)
    }

    #[instrument(level = "debug", skip(self), ret)]
    fn compute_external_query_constraints(&self) -> Result<ExternalConstraints<'tcx>, NoSolution> {
        // Cannot use `take_registered_region_obligations` as we may compute the response
        // inside of a `probe` whenever we have multiple choices inside of the solver.
        let region_obligations = self.infcx.inner.borrow().region_obligations().to_owned();
        let region_constraints = self.infcx.with_region_constraints(|region_constraints| {
            make_query_region_constraints(
                self.tcx(),
                region_obligations
                    .iter()
                    .map(|r_o| (r_o.sup_type, r_o.sub_region, r_o.origin.to_constraint_category())),
                region_constraints,
            )
        });
        let opaque_types = self.infcx.clone_opaque_types_for_query_response();
        Ok(self
            .tcx()
            .mk_external_constraints(ExternalConstraintsData { region_constraints, opaque_types }))
    }

    /// After calling a canonical query, we apply the constraints returned
    /// by the query using this function.
    ///
    /// This happens in three steps:
    /// - we instantiate the bound variables of the query response
    /// - we unify the `var_values` of the response with the `original_values`
    /// - we apply the `external_constraints` returned by the query
    pub(super) fn instantiate_and_apply_query_response(
        &mut self,
        param_env: ty::ParamEnv<'tcx>,
        original_values: Vec<ty::GenericArg<'tcx>>,
        response: CanonicalResponse<'tcx>,
    ) -> Result<(Certainty, Vec<Goal<'tcx, ty::Predicate<'tcx>>>), NoSolution> {
        let substitution = self.compute_query_response_substitution(&original_values, &response);

        let Response { var_values, external_constraints, certainty } =
            response.substitute(self.tcx(), &substitution);

        let nested_goals = self.unify_query_var_values(param_env, &original_values, var_values)?;

        // FIXME: implement external constraints.
        let ExternalConstraintsData { region_constraints, opaque_types: _ } =
            external_constraints.deref();
        self.register_region_constraints(region_constraints);

        Ok((certainty, nested_goals))
    }

    /// This returns the substitutions to instantiate the bound variables of
    /// the canonical reponse. This depends on the `original_values` for the
    /// bound variables.
    fn compute_query_response_substitution(
        &self,
        original_values: &[ty::GenericArg<'tcx>],
        response: &CanonicalResponse<'tcx>,
    ) -> CanonicalVarValues<'tcx> {
        // FIXME: Longterm canonical queries should deal with all placeholders
        // created inside of the query directly instead of returning them to the
        // caller.
        let prev_universe = self.infcx.universe();
        let universes_created_in_query = response.max_universe.index() + 1;
        for _ in 0..universes_created_in_query {
            self.infcx.create_next_universe();
        }

        let var_values = response.value.var_values;
        assert_eq!(original_values.len(), var_values.len());

        // If the query did not make progress with constraining inference variables,
        // we would normally create a new inference variables for bound existential variables
        // only then unify this new inference variable with the inference variable from
        // the input.
        //
        // We therefore instantiate the existential variable in the canonical response with the
        // inference variable of the input right away, which is more performant.
        let mut opt_values = vec![None; response.variables.len()];
        for (original_value, result_value) in iter::zip(original_values, var_values.var_values) {
            match result_value.unpack() {
                GenericArgKind::Type(t) => {
                    if let &ty::Bound(debruijn, b) = t.kind() {
                        assert_eq!(debruijn, ty::INNERMOST);
                        opt_values[b.var.index()] = Some(*original_value);
                    }
                }
                GenericArgKind::Lifetime(r) => {
                    if let ty::ReLateBound(debruijn, br) = *r {
                        assert_eq!(debruijn, ty::INNERMOST);
                        opt_values[br.var.index()] = Some(*original_value);
                    }
                }
                GenericArgKind::Const(c) => {
                    if let ty::ConstKind::Bound(debrujin, b) = c.kind() {
                        assert_eq!(debrujin, ty::INNERMOST);
                        opt_values[b.index()] = Some(*original_value);
                    }
                }
            }
        }

        let var_values = self.tcx().mk_substs_from_iter(response.variables.iter().enumerate().map(
            |(index, info)| {
                if info.universe() != ty::UniverseIndex::ROOT {
                    // A variable from inside a binder of the query. While ideally these shouldn't
                    // exist at all (see the FIXME at the start of this method), we have to deal with
                    // them for now.
                    self.infcx.instantiate_canonical_var(DUMMY_SP, info, |idx| {
                        ty::UniverseIndex::from(prev_universe.index() + idx.index())
                    })
                } else if info.is_existential() {
                    // As an optimization we sometimes avoid creating a new inference variable here.
                    //
                    // All new inference variables we create start out in the current universe of the caller.
                    // This is conceptionally wrong as these inference variables would be able to name
                    // more placeholders then they should be able to. However the inference variables have
                    // to "come from somewhere", so by equating them with the original values of the caller
                    // later on, we pull them down into their correct universe again.
                    if let Some(v) = opt_values[index] {
                        v
                    } else {
                        self.infcx.instantiate_canonical_var(DUMMY_SP, info, |_| prev_universe)
                    }
                } else {
                    // For placeholders which were already part of the input, we simply map this
                    // universal bound variable back the placeholder of the input.
                    original_values[info.expect_anon_placeholder() as usize]
                }
            },
        ));

        CanonicalVarValues { var_values }
    }

    #[instrument(level = "debug", skip(self, param_env), ret)]
    fn unify_query_var_values(
        &self,
        param_env: ty::ParamEnv<'tcx>,
        original_values: &[ty::GenericArg<'tcx>],
        var_values: CanonicalVarValues<'tcx>,
    ) -> Result<Vec<Goal<'tcx, ty::Predicate<'tcx>>>, NoSolution> {
        assert_eq!(original_values.len(), var_values.len());

        let mut nested_goals = vec![];
        for (&orig, response) in iter::zip(original_values, var_values.var_values) {
            nested_goals.extend(self.eq_and_get_goals(param_env, orig, response)?);
        }

        Ok(nested_goals)
    }

    fn register_region_constraints(&mut self, region_constraints: &QueryRegionConstraints<'tcx>) {
        for &(ty::OutlivesPredicate(lhs, rhs), _) in &region_constraints.outlives {
            match lhs.unpack() {
                GenericArgKind::Lifetime(lhs) => self.register_region_outlives(lhs, rhs),
                GenericArgKind::Type(lhs) => self.register_ty_outlives(lhs, rhs),
                GenericArgKind::Const(_) => bug!("const outlives: {lhs:?}: {rhs:?}"),
            }
        }

        for member_constraint in &region_constraints.member_constraints {
            // FIXME: Deal with member constraints :<
            let _ = member_constraint;
        }
    }
}
