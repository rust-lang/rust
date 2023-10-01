//! Canonicalization is used to separate some goal from its context,
//! throwing away unnecessary information in the process.
//!
//! This is necessary to cache goals containing inference variables
//! and placeholders without restricting them to the current `InferCtxt`.
//!
//! Canonicalization is fairly involved, for more details see the relevant
//! section of the [rustc-dev-guide][c].
//!
//! [c]: https://rustc-dev-guide.rust-lang.org/solve/canonicalization.html
use super::{CanonicalInput, Certainty, EvalCtxt, Goal};
use crate::solve::canonicalize::{CanonicalizeMode, Canonicalizer};
use crate::solve::{
    inspect, response_no_constraints_raw, CanonicalResponse, QueryResult, Response,
};
use rustc_data_structures::fx::FxHashSet;
use rustc_index::IndexVec;
use rustc_infer::infer::canonical::query_response::make_query_region_constraints;
use rustc_infer::infer::canonical::CanonicalVarValues;
use rustc_infer::infer::canonical::{CanonicalExt, QueryRegionConstraints};
use rustc_infer::infer::{DefineOpaqueTypes, InferCtxt, InferOk};
use rustc_middle::infer::canonical::Canonical;
use rustc_middle::traits::query::NoSolution;
use rustc_middle::traits::solve::{
    ExternalConstraintsData, MaybeCause, PredefinedOpaquesData, QueryInput,
};
use rustc_middle::traits::ObligationCause;
use rustc_middle::ty::{
    self, BoundVar, GenericArgKind, Ty, TyCtxt, TypeFoldable, TypeFolder, TypeSuperFoldable,
    TypeVisitableExt,
};
use rustc_span::DUMMY_SP;
use std::iter;
use std::ops::Deref;

trait ResponseT<'tcx> {
    fn var_values(&self) -> CanonicalVarValues<'tcx>;
}

impl<'tcx> ResponseT<'tcx> for Response<'tcx> {
    fn var_values(&self) -> CanonicalVarValues<'tcx> {
        self.var_values
    }
}

impl<'tcx, T> ResponseT<'tcx> for inspect::State<'tcx, T> {
    fn var_values(&self) -> CanonicalVarValues<'tcx> {
        self.var_values
    }
}

impl<'tcx> EvalCtxt<'_, 'tcx> {
    /// Canonicalizes the goal remembering the original values
    /// for each bound variable.
    pub(super) fn canonicalize_goal<T: TypeFoldable<TyCtxt<'tcx>>>(
        &self,
        goal: Goal<'tcx, T>,
    ) -> (Vec<ty::GenericArg<'tcx>>, CanonicalInput<'tcx, T>) {
        let opaque_types = self.infcx.clone_opaque_types_for_query_response();
        let (goal, opaque_types) =
            (goal, opaque_types).fold_with(&mut EagerResolver { infcx: self.infcx });

        let mut orig_values = Default::default();
        let canonical_goal = Canonicalizer::canonicalize(
            self.infcx,
            CanonicalizeMode::Input,
            &mut orig_values,
            QueryInput {
                goal,
                anchor: self.infcx.defining_use_anchor,
                predefined_opaques_in_body: self
                    .tcx()
                    .mk_predefined_opaques_in_body(PredefinedOpaquesData { opaque_types }),
            },
        );
        (orig_values, canonical_goal)
    }

    /// To return the constraints of a canonical query to the caller, we canonicalize:
    ///
    /// - `var_values`: a map from bound variables in the canonical goal to
    ///   the values inferred while solving the instantiated goal.
    /// - `external_constraints`: additional constraints which aren't expressible
    ///   using simple unification of inference variables.
    #[instrument(level = "debug", skip(self))]
    pub(in crate::solve) fn evaluate_added_goals_and_make_canonical_response(
        &mut self,
        certainty: Certainty,
    ) -> QueryResult<'tcx> {
        let goals_certainty = self.try_evaluate_added_goals()?;
        assert_eq!(
            self.tainted,
            Ok(()),
            "EvalCtxt is tainted -- nested goals may have been dropped in a \
            previous call to `try_evaluate_added_goals!`"
        );

        let certainty = certainty.unify_with(goals_certainty);
        if let Certainty::OVERFLOW = certainty {
            // If we have overflow, it's probable that we're substituting a type
            // into itself infinitely and any partial substitutions in the query
            // response are probably not useful anyways, so just return an empty
            // query response.
            //
            // This may prevent us from potentially useful inference, e.g.
            // 2 candidates, one ambiguous and one overflow, which both
            // have the same inference constraints.
            //
            // Changing this to retain some constraints in the future
            // won't be a breaking change, so this is good enough for now.
            return Ok(self.make_ambiguous_response_no_constraints(MaybeCause::Overflow));
        }

        let var_values = self.var_values;
        let external_constraints = self.compute_external_query_constraints()?;

        let (var_values, mut external_constraints) =
            (var_values, external_constraints).fold_with(&mut EagerResolver { infcx: self.infcx });
        // Remove any trivial region constraints once we've resolved regions
        external_constraints
            .region_constraints
            .outlives
            .retain(|(outlives, _)| outlives.0.as_region().map_or(true, |re| re != outlives.1));

        let canonical = Canonicalizer::canonicalize(
            self.infcx,
            CanonicalizeMode::Response { max_input_universe: self.max_input_universe },
            &mut Default::default(),
            Response {
                var_values,
                certainty,
                external_constraints: self.tcx().mk_external_constraints(external_constraints),
            },
        );

        Ok(canonical)
    }

    /// Constructs a totally unconstrained, ambiguous response to a goal.
    ///
    /// Take care when using this, since often it's useful to respond with
    /// ambiguity but return constrained variables to guide inference.
    pub(in crate::solve) fn make_ambiguous_response_no_constraints(
        &self,
        maybe_cause: MaybeCause,
    ) -> CanonicalResponse<'tcx> {
        response_no_constraints_raw(
            self.tcx(),
            self.max_input_universe,
            self.variables,
            Certainty::Maybe(maybe_cause),
        )
    }

    /// Computes the region constraints and *new* opaque types registered when
    /// proving a goal.
    ///
    /// If an opaque was already constrained before proving this goal, then the
    /// external constraints do not need to record that opaque, since if it is
    /// further constrained by inference, that will be passed back in the var
    /// values.
    #[instrument(level = "debug", skip(self), ret)]
    fn compute_external_query_constraints(
        &self,
    ) -> Result<ExternalConstraintsData<'tcx>, NoSolution> {
        // We only check for leaks from universes which were entered inside
        // of the query.
        self.infcx.leak_check(self.max_input_universe, None).map_err(|e| {
            debug!(?e, "failed the leak check");
            NoSolution
        })?;

        // Cannot use `take_registered_region_obligations` as we may compute the response
        // inside of a `probe` whenever we have multiple choices inside of the solver.
        let region_obligations = self.infcx.inner.borrow().region_obligations().to_owned();
        let mut region_constraints = self.infcx.with_region_constraints(|region_constraints| {
            make_query_region_constraints(
                self.tcx(),
                region_obligations
                    .iter()
                    .map(|r_o| (r_o.sup_type, r_o.sub_region, r_o.origin.to_constraint_category())),
                region_constraints,
            )
        });

        let mut seen = FxHashSet::default();
        region_constraints.outlives.retain(|outlives| seen.insert(*outlives));

        let mut opaque_types = self.infcx.clone_opaque_types_for_query_response();
        // Only return opaque type keys for newly-defined opaques
        opaque_types.retain(|(a, _)| {
            self.predefined_opaques_in_body.opaque_types.iter().all(|(pa, _)| pa != a)
        });

        Ok(ExternalConstraintsData { region_constraints, opaque_types })
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
        let substitution =
            Self::compute_query_response_substitution(self.infcx, &original_values, &response);

        let Response { var_values, external_constraints, certainty } =
            response.substitute(self.tcx(), &substitution);

        let nested_goals =
            Self::unify_query_var_values(self.infcx, param_env, &original_values, var_values)?;

        let ExternalConstraintsData { region_constraints, opaque_types } =
            external_constraints.deref();
        self.register_region_constraints(region_constraints);
        self.register_opaque_types(param_env, opaque_types)?;

        Ok((certainty, nested_goals))
    }

    /// This returns the substitutions to instantiate the bound variables of
    /// the canonical response. This depends on the `original_values` for the
    /// bound variables.
    fn compute_query_response_substitution<T: ResponseT<'tcx>>(
        infcx: &InferCtxt<'tcx>,
        original_values: &[ty::GenericArg<'tcx>],
        response: &Canonical<'tcx, T>,
    ) -> CanonicalVarValues<'tcx> {
        // FIXME: Longterm canonical queries should deal with all placeholders
        // created inside of the query directly instead of returning them to the
        // caller.
        let prev_universe = infcx.universe();
        let universes_created_in_query = response.max_universe.index();
        for _ in 0..universes_created_in_query {
            infcx.create_next_universe();
        }

        let var_values = response.value.var_values();
        assert_eq!(original_values.len(), var_values.len());

        // If the query did not make progress with constraining inference variables,
        // we would normally create a new inference variables for bound existential variables
        // only then unify this new inference variable with the inference variable from
        // the input.
        //
        // We therefore instantiate the existential variable in the canonical response with the
        // inference variable of the input right away, which is more performant.
        let mut opt_values = IndexVec::from_elem_n(None, response.variables.len());
        for (original_value, result_value) in iter::zip(original_values, var_values.var_values) {
            match result_value.unpack() {
                GenericArgKind::Type(t) => {
                    if let &ty::Bound(debruijn, b) = t.kind() {
                        assert_eq!(debruijn, ty::INNERMOST);
                        opt_values[b.var] = Some(*original_value);
                    }
                }
                GenericArgKind::Lifetime(r) => {
                    if let ty::ReLateBound(debruijn, br) = *r {
                        assert_eq!(debruijn, ty::INNERMOST);
                        opt_values[br.var] = Some(*original_value);
                    }
                }
                GenericArgKind::Const(c) => {
                    if let ty::ConstKind::Bound(debruijn, b) = c.kind() {
                        assert_eq!(debruijn, ty::INNERMOST);
                        opt_values[b] = Some(*original_value);
                    }
                }
            }
        }

        let var_values = infcx.tcx.mk_args_from_iter(response.variables.iter().enumerate().map(
            |(index, info)| {
                if info.universe() != ty::UniverseIndex::ROOT {
                    // A variable from inside a binder of the query. While ideally these shouldn't
                    // exist at all (see the FIXME at the start of this method), we have to deal with
                    // them for now.
                    infcx.instantiate_canonical_var(DUMMY_SP, info, |idx| {
                        ty::UniverseIndex::from(prev_universe.index() + idx.index())
                    })
                } else if info.is_existential() {
                    // As an optimization we sometimes avoid creating a new inference variable here.
                    //
                    // All new inference variables we create start out in the current universe of the caller.
                    // This is conceptually wrong as these inference variables would be able to name
                    // more placeholders then they should be able to. However the inference variables have
                    // to "come from somewhere", so by equating them with the original values of the caller
                    // later on, we pull them down into their correct universe again.
                    if let Some(v) = opt_values[BoundVar::from_usize(index)] {
                        v
                    } else {
                        infcx.instantiate_canonical_var(DUMMY_SP, info, |_| prev_universe)
                    }
                } else {
                    // For placeholders which were already part of the input, we simply map this
                    // universal bound variable back the placeholder of the input.
                    original_values[info.expect_placeholder_index()]
                }
            },
        ));

        CanonicalVarValues { var_values }
    }

    #[instrument(level = "debug", skip(infcx, param_env), ret)]
    fn unify_query_var_values(
        infcx: &InferCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        original_values: &[ty::GenericArg<'tcx>],
        var_values: CanonicalVarValues<'tcx>,
    ) -> Result<Vec<Goal<'tcx, ty::Predicate<'tcx>>>, NoSolution> {
        assert_eq!(original_values.len(), var_values.len());

        let mut nested_goals = vec![];
        for (&orig, response) in iter::zip(original_values, var_values.var_values) {
            nested_goals.extend(
                infcx
                    .at(&ObligationCause::dummy(), param_env)
                    .eq(DefineOpaqueTypes::No, orig, response)
                    .map(|InferOk { value: (), obligations }| {
                        obligations.into_iter().map(|o| Goal::from(o))
                    })
                    .map_err(|e| {
                        debug!(?e, "failed to equate");
                        NoSolution
                    })?,
            );
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

    fn register_opaque_types(
        &mut self,
        param_env: ty::ParamEnv<'tcx>,
        opaque_types: &[(ty::OpaqueTypeKey<'tcx>, Ty<'tcx>)],
    ) -> Result<(), NoSolution> {
        for &(key, ty) in opaque_types {
            self.insert_hidden_type(key, param_env, ty)?;
        }
        Ok(())
    }
}

/// Resolves ty, region, and const vars to their inferred values or their root vars.
struct EagerResolver<'a, 'tcx> {
    infcx: &'a InferCtxt<'tcx>,
}

impl<'tcx> TypeFolder<TyCtxt<'tcx>> for EagerResolver<'_, 'tcx> {
    fn interner(&self) -> TyCtxt<'tcx> {
        self.infcx.tcx
    }

    fn fold_ty(&mut self, t: Ty<'tcx>) -> Ty<'tcx> {
        match *t.kind() {
            ty::Infer(ty::TyVar(vid)) => match self.infcx.probe_ty_var(vid) {
                Ok(t) => t.fold_with(self),
                Err(_) => Ty::new_var(self.infcx.tcx, self.infcx.root_var(vid)),
            },
            ty::Infer(ty::IntVar(vid)) => self.infcx.opportunistic_resolve_int_var(vid),
            ty::Infer(ty::FloatVar(vid)) => self.infcx.opportunistic_resolve_float_var(vid),
            _ => {
                if t.has_infer() {
                    t.super_fold_with(self)
                } else {
                    t
                }
            }
        }
    }

    fn fold_region(&mut self, r: ty::Region<'tcx>) -> ty::Region<'tcx> {
        match *r {
            ty::ReVar(vid) => self
                .infcx
                .inner
                .borrow_mut()
                .unwrap_region_constraints()
                .opportunistic_resolve_var(self.infcx.tcx, vid),
            _ => r,
        }
    }

    fn fold_const(&mut self, c: ty::Const<'tcx>) -> ty::Const<'tcx> {
        match c.kind() {
            ty::ConstKind::Infer(ty::InferConst::Var(vid)) => {
                // FIXME: we need to fold the ty too, I think.
                match self.infcx.probe_const_var(vid) {
                    Ok(c) => c.fold_with(self),
                    Err(_) => {
                        ty::Const::new_var(self.infcx.tcx, self.infcx.root_const_var(vid), c.ty())
                    }
                }
            }
            ty::ConstKind::Infer(ty::InferConst::EffectVar(vid)) => {
                debug_assert_eq!(c.ty(), self.infcx.tcx.types.bool);
                match self.infcx.probe_effect_var(vid) {
                    Some(c) => c.as_const(self.infcx.tcx),
                    None => ty::Const::new_infer(
                        self.infcx.tcx,
                        ty::InferConst::EffectVar(self.infcx.root_effect_var(vid)),
                        self.infcx.tcx.types.bool,
                    ),
                }
            }
            _ => {
                if c.has_infer() {
                    c.super_fold_with(self)
                } else {
                    c
                }
            }
        }
    }
}

impl<'tcx> inspect::ProofTreeBuilder<'tcx> {
    pub fn make_canonical_state<T: TypeFoldable<TyCtxt<'tcx>>>(
        ecx: &EvalCtxt<'_, 'tcx>,
        data: T,
    ) -> inspect::CanonicalState<'tcx, T> {
        let state = inspect::State { var_values: ecx.var_values, data };
        let state = state.fold_with(&mut EagerResolver { infcx: ecx.infcx });
        Canonicalizer::canonicalize(
            ecx.infcx,
            CanonicalizeMode::Response { max_input_universe: ecx.max_input_universe },
            &mut vec![],
            state,
        )
    }

    pub fn instantiate_canonical_state<T: TypeFoldable<TyCtxt<'tcx>>>(
        infcx: &InferCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        original_values: &[ty::GenericArg<'tcx>],
        state: inspect::CanonicalState<'tcx, T>,
    ) -> Result<(Vec<Goal<'tcx, ty::Predicate<'tcx>>>, T), NoSolution> {
        let substitution =
            EvalCtxt::compute_query_response_substitution(infcx, original_values, &state);

        let inspect::State { var_values, data } = state.substitute(infcx.tcx, &substitution);

        let nested_goals =
            EvalCtxt::unify_query_var_values(infcx, param_env, original_values, var_values)?;
        Ok((nested_goals, data))
    }
}
