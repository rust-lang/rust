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

use std::iter;

use rustc_index::IndexVec;
use rustc_type_ir::data_structures::HashSet;
use rustc_type_ir::inherent::*;
use rustc_type_ir::relate::solver_relating::RelateExt;
use rustc_type_ir::{
    self as ty, Canonical, CanonicalVarValues, InferCtxtLike, Interner, TypeFoldable,
};
use tracing::{debug, instrument, trace};

use crate::canonicalizer::Canonicalizer;
use crate::delegate::SolverDelegate;
use crate::resolve::EagerResolver;
use crate::solve::eval_ctxt::CurrentGoalKind;
use crate::solve::{
    CanonicalInput, CanonicalResponse, Certainty, EvalCtxt, ExternalConstraintsData, Goal,
    MaybeCause, NestedNormalizationGoals, NoSolution, PredefinedOpaquesData, QueryInput,
    QueryResult, Response, inspect, response_no_constraints_raw,
};

trait ResponseT<I: Interner> {
    fn var_values(&self) -> CanonicalVarValues<I>;
}

impl<I: Interner> ResponseT<I> for Response<I> {
    fn var_values(&self) -> CanonicalVarValues<I> {
        self.var_values
    }
}

impl<I: Interner, T> ResponseT<I> for inspect::State<I, T> {
    fn var_values(&self) -> CanonicalVarValues<I> {
        self.var_values
    }
}

impl<D, I> EvalCtxt<'_, D>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    /// Canonicalizes the goal remembering the original values
    /// for each bound variable.
    pub(super) fn canonicalize_goal<T: TypeFoldable<I>>(
        &self,
        goal: Goal<I, T>,
    ) -> (Vec<I::GenericArg>, CanonicalInput<I, T>) {
        // We only care about one entry per `OpaqueTypeKey` here,
        // so we only canonicalize the lookup table and ignore
        // duplicate entries.
        let opaque_types = self.delegate.clone_opaque_types_lookup_table();
        let (goal, opaque_types) =
            (goal, opaque_types).fold_with(&mut EagerResolver::new(self.delegate));

        let mut orig_values = Default::default();
        let canonical = Canonicalizer::canonicalize_input(
            self.delegate,
            &mut orig_values,
            QueryInput {
                goal,
                predefined_opaques_in_body: self
                    .cx()
                    .mk_predefined_opaques_in_body(PredefinedOpaquesData { opaque_types }),
            },
        );
        let query_input = ty::CanonicalQueryInput { canonical, typing_mode: self.typing_mode() };
        (orig_values, query_input)
    }

    /// To return the constraints of a canonical query to the caller, we canonicalize:
    ///
    /// - `var_values`: a map from bound variables in the canonical goal to
    ///   the values inferred while solving the instantiated goal.
    /// - `external_constraints`: additional constraints which aren't expressible
    ///   using simple unification of inference variables.
    ///
    /// This takes the `shallow_certainty` which represents whether we're confident
    /// that the final result of the current goal only depends on the nested goals.
    ///
    /// In case this is `Certainy::Maybe`, there may still be additional nested goals
    /// or inference constraints required for this candidate to be hold. The candidate
    /// always requires all already added constraints and nested goals.
    #[instrument(level = "trace", skip(self), ret)]
    pub(in crate::solve) fn evaluate_added_goals_and_make_canonical_response(
        &mut self,
        shallow_certainty: Certainty,
    ) -> QueryResult<I> {
        self.inspect.make_canonical_response(shallow_certainty);

        let goals_certainty = self.try_evaluate_added_goals()?;
        assert_eq!(
            self.tainted,
            Ok(()),
            "EvalCtxt is tainted -- nested goals may have been dropped in a \
            previous call to `try_evaluate_added_goals!`"
        );

        // We only check for leaks from universes which were entered inside
        // of the query.
        self.delegate.leak_check(self.max_input_universe).map_err(|NoSolution| {
            trace!("failed the leak check");
            NoSolution
        })?;

        let (certainty, normalization_nested_goals) =
            match (self.current_goal_kind, shallow_certainty) {
                // When normalizing, we've replaced the expected term with an unconstrained
                // inference variable. This means that we dropped information which could
                // have been important. We handle this by instead returning the nested goals
                // to the caller, where they are then handled. We only do so if we do not
                // need to recompute the `NormalizesTo` goal afterwards to avoid repeatedly
                // uplifting its nested goals. This is the case if the `shallow_certainty` is
                // `Certainty::Yes`.
                (CurrentGoalKind::NormalizesTo, Certainty::Yes) => {
                    let goals = std::mem::take(&mut self.nested_goals);
                    // As we return all ambiguous nested goals, we can ignore the certainty
                    // returned by `self.try_evaluate_added_goals()`.
                    if goals.is_empty() {
                        assert!(matches!(goals_certainty, Certainty::Yes));
                    }
                    (Certainty::Yes, NestedNormalizationGoals(goals))
                }
                _ => {
                    let certainty = shallow_certainty.and(goals_certainty);
                    (certainty, NestedNormalizationGoals::empty())
                }
            };

        if let Certainty::Maybe(cause @ MaybeCause::Overflow { keep_constraints: false, .. }) =
            certainty
        {
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
            return Ok(self.make_ambiguous_response_no_constraints(cause));
        }

        let external_constraints =
            self.compute_external_query_constraints(certainty, normalization_nested_goals);
        let (var_values, mut external_constraints) = (self.var_values, external_constraints)
            .fold_with(&mut EagerResolver::new(self.delegate));

        // Remove any trivial or duplicated region constraints once we've resolved regions
        let mut unique = HashSet::default();
        external_constraints.region_constraints.retain(|outlives| {
            outlives.0.as_region().is_none_or(|re| re != outlives.1) && unique.insert(*outlives)
        });

        let canonical = Canonicalizer::canonicalize_response(
            self.delegate,
            self.max_input_universe,
            &mut Default::default(),
            Response {
                var_values,
                certainty,
                external_constraints: self.cx().mk_external_constraints(external_constraints),
            },
        );

        // HACK: We bail with overflow if the response would have too many non-region
        // inference variables. This tends to only happen if we encounter a lot of
        // ambiguous alias types which get replaced with fresh inference variables
        // during generalization. This prevents hangs caused by an exponential blowup,
        // see tests/ui/traits/next-solver/coherence-alias-hang.rs.
        match self.current_goal_kind {
            // We don't do so for `NormalizesTo` goals as we erased the expected term and
            // bailing with overflow here would prevent us from detecting a type-mismatch,
            // causing a coherence error in diesel, see #131969. We still bail with overflow
            // when later returning from the parent AliasRelate goal.
            CurrentGoalKind::NormalizesTo => {}
            CurrentGoalKind::Misc | CurrentGoalKind::CoinductiveTrait => {
                let num_non_region_vars = canonical
                    .variables
                    .iter()
                    .filter(|c| !c.is_region() && c.is_existential())
                    .count();
                if num_non_region_vars > self.cx().recursion_limit() {
                    debug!(?num_non_region_vars, "too many inference variables -> overflow");
                    return Ok(self.make_ambiguous_response_no_constraints(MaybeCause::Overflow {
                        suggest_increasing_limit: true,
                        keep_constraints: false,
                    }));
                }
            }
        }

        Ok(canonical)
    }

    /// Constructs a totally unconstrained, ambiguous response to a goal.
    ///
    /// Take care when using this, since often it's useful to respond with
    /// ambiguity but return constrained variables to guide inference.
    pub(in crate::solve) fn make_ambiguous_response_no_constraints(
        &self,
        maybe_cause: MaybeCause,
    ) -> CanonicalResponse<I> {
        response_no_constraints_raw(
            self.cx(),
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
    #[instrument(level = "trace", skip(self), ret)]
    fn compute_external_query_constraints(
        &self,
        certainty: Certainty,
        normalization_nested_goals: NestedNormalizationGoals<I>,
    ) -> ExternalConstraintsData<I> {
        // We only return region constraints once the certainty is `Yes`. This
        // is necessary as we may drop nested goals on ambiguity, which may result
        // in unconstrained inference variables in the region constraints. It also
        // prevents us from emitting duplicate region constraints, avoiding some
        // unnecessary work. This slightly weakens the leak check in case it uses
        // region constraints from an ambiguous nested goal. This is tested in both
        // `tests/ui/higher-ranked/leak-check/leak-check-in-selection-5-ambig.rs` and
        // `tests/ui/higher-ranked/leak-check/leak-check-in-selection-6-ambig-unify.rs`.
        let region_constraints = if certainty == Certainty::Yes {
            self.delegate.make_deduplicated_outlives_constraints()
        } else {
            Default::default()
        };

        // We only return *newly defined* opaque types from canonical queries.
        //
        // Constraints for any existing opaque types are already tracked by changes
        // to the `var_values`.
        let opaque_types = self
            .delegate
            .clone_opaque_types_added_since(self.initial_opaque_types_storage_num_entries);

        ExternalConstraintsData { region_constraints, opaque_types, normalization_nested_goals }
    }

    /// After calling a canonical query, we apply the constraints returned
    /// by the query using this function.
    ///
    /// This happens in three steps:
    /// - we instantiate the bound variables of the query response
    /// - we unify the `var_values` of the response with the `original_values`
    /// - we apply the `external_constraints` returned by the query, returning
    ///   the `normalization_nested_goals`
    pub(super) fn instantiate_and_apply_query_response(
        &mut self,
        param_env: I::ParamEnv,
        original_values: Vec<I::GenericArg>,
        response: CanonicalResponse<I>,
    ) -> (NestedNormalizationGoals<I>, Certainty) {
        let instantiation = Self::compute_query_response_instantiation_values(
            self.delegate,
            &original_values,
            &response,
            self.origin_span,
        );

        let Response { var_values, external_constraints, certainty } =
            self.delegate.instantiate_canonical(response, instantiation);

        Self::unify_query_var_values(
            self.delegate,
            param_env,
            &original_values,
            var_values,
            self.origin_span,
        );

        let ExternalConstraintsData {
            region_constraints,
            opaque_types,
            normalization_nested_goals,
        } = &*external_constraints;

        self.register_region_constraints(region_constraints);
        self.register_new_opaque_types(opaque_types);

        (normalization_nested_goals.clone(), certainty)
    }

    /// This returns the canonical variable values to instantiate the bound variables of
    /// the canonical response. This depends on the `original_values` for the
    /// bound variables.
    fn compute_query_response_instantiation_values<T: ResponseT<I>>(
        delegate: &D,
        original_values: &[I::GenericArg],
        response: &Canonical<I, T>,
        span: I::Span,
    ) -> CanonicalVarValues<I> {
        // FIXME: Longterm canonical queries should deal with all placeholders
        // created inside of the query directly instead of returning them to the
        // caller.
        let prev_universe = delegate.universe();
        let universes_created_in_query = response.max_universe.index();
        for _ in 0..universes_created_in_query {
            delegate.create_next_universe();
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
        for (original_value, result_value) in
            iter::zip(original_values, var_values.var_values.iter())
        {
            match result_value.kind() {
                ty::GenericArgKind::Type(t) => {
                    if let ty::Bound(debruijn, b) = t.kind() {
                        assert_eq!(debruijn, ty::INNERMOST);
                        opt_values[b.var()] = Some(*original_value);
                    }
                }
                ty::GenericArgKind::Lifetime(r) => {
                    if let ty::ReBound(debruijn, br) = r.kind() {
                        assert_eq!(debruijn, ty::INNERMOST);
                        opt_values[br.var()] = Some(*original_value);
                    }
                }
                ty::GenericArgKind::Const(c) => {
                    if let ty::ConstKind::Bound(debruijn, bv) = c.kind() {
                        assert_eq!(debruijn, ty::INNERMOST);
                        opt_values[bv.var()] = Some(*original_value);
                    }
                }
            }
        }

        let var_values = delegate.cx().mk_args_from_iter(
            response.variables.iter().enumerate().map(|(index, info)| {
                if info.universe() != ty::UniverseIndex::ROOT {
                    // A variable from inside a binder of the query. While ideally these shouldn't
                    // exist at all (see the FIXME at the start of this method), we have to deal with
                    // them for now.
                    delegate.instantiate_canonical_var_with_infer(info, span, |idx| {
                        prev_universe + idx.index()
                    })
                } else if info.is_existential() {
                    // As an optimization we sometimes avoid creating a new inference variable here.
                    //
                    // All new inference variables we create start out in the current universe of the caller.
                    // This is conceptually wrong as these inference variables would be able to name
                    // more placeholders then they should be able to. However the inference variables have
                    // to "come from somewhere", so by equating them with the original values of the caller
                    // later on, we pull them down into their correct universe again.
                    if let Some(v) = opt_values[ty::BoundVar::from_usize(index)] {
                        v
                    } else {
                        delegate.instantiate_canonical_var_with_infer(info, span, |_| prev_universe)
                    }
                } else {
                    // For placeholders which were already part of the input, we simply map this
                    // universal bound variable back the placeholder of the input.
                    original_values[info.expect_placeholder_index()]
                }
            }),
        );

        CanonicalVarValues { var_values }
    }

    /// Unify the `original_values` with the `var_values` returned by the canonical query..
    ///
    /// This assumes that this unification will always succeed. This is the case when
    /// applying a query response right away. However, calling a canonical query, doing any
    /// other kind of trait solving, and only then instantiating the result of the query
    /// can cause the instantiation to fail. This is not supported and we ICE in this case.
    ///
    /// We always structurally instantiate aliases. Relating aliases needs to be different
    /// depending on whether the alias is *rigid* or not. We're only really able to tell
    /// whether an alias is rigid by using the trait solver. When instantiating a response
    /// from the solver we assume that the solver correctly handled aliases and therefore
    /// always relate them structurally here.
    #[instrument(level = "trace", skip(delegate))]
    fn unify_query_var_values(
        delegate: &D,
        param_env: I::ParamEnv,
        original_values: &[I::GenericArg],
        var_values: CanonicalVarValues<I>,
        span: I::Span,
    ) {
        assert_eq!(original_values.len(), var_values.len());

        for (&orig, response) in iter::zip(original_values, var_values.var_values.iter()) {
            let goals =
                delegate.eq_structurally_relating_aliases(param_env, orig, response, span).unwrap();
            assert!(goals.is_empty());
        }
    }

    fn register_region_constraints(
        &mut self,
        outlives: &[ty::OutlivesPredicate<I, I::GenericArg>],
    ) {
        for &ty::OutlivesPredicate(lhs, rhs) in outlives {
            match lhs.kind() {
                ty::GenericArgKind::Lifetime(lhs) => self.register_region_outlives(lhs, rhs),
                ty::GenericArgKind::Type(lhs) => self.register_ty_outlives(lhs, rhs),
                ty::GenericArgKind::Const(_) => panic!("const outlives: {lhs:?}: {rhs:?}"),
            }
        }
    }

    fn register_new_opaque_types(&mut self, opaque_types: &[(ty::OpaqueTypeKey<I>, I::Ty)]) {
        for &(key, ty) in opaque_types {
            let prev = self.delegate.register_hidden_type_in_storage(key, ty, self.origin_span);
            // We eagerly resolve inference variables when computing the query response.
            // This can cause previously distinct opaque type keys to now be structurally equal.
            //
            // To handle this, we store any duplicate entries in a separate list to check them
            // at the end of typeck/borrowck. We could alternatively eagerly equate the hidden
            // types here. However, doing so is difficult as it may result in nested goals and
            // any errors may make it harder to track the control flow for diagnostics.
            if let Some(prev) = prev {
                self.delegate.add_duplicate_opaque_type(key, prev, self.origin_span);
            }
        }
    }
}

/// Used by proof trees to be able to recompute intermediate actions while
/// evaluating a goal. The `var_values` not only include the bound variables
/// of the query input, but also contain all unconstrained inference vars
/// created while evaluating this goal.
pub(in crate::solve) fn make_canonical_state<D, T, I>(
    delegate: &D,
    var_values: &[I::GenericArg],
    max_input_universe: ty::UniverseIndex,
    data: T,
) -> inspect::CanonicalState<I, T>
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
    T: TypeFoldable<I>,
{
    let var_values = CanonicalVarValues { var_values: delegate.cx().mk_args(var_values) };
    let state = inspect::State { var_values, data };
    let state = state.fold_with(&mut EagerResolver::new(delegate));
    Canonicalizer::canonicalize_response(delegate, max_input_universe, &mut vec![], state)
}

// FIXME: needs to be pub to be accessed by downstream
// `rustc_trait_selection::solve::inspect::analyse`.
pub fn instantiate_canonical_state<D, I, T: TypeFoldable<I>>(
    delegate: &D,
    span: I::Span,
    param_env: I::ParamEnv,
    orig_values: &mut Vec<I::GenericArg>,
    state: inspect::CanonicalState<I, T>,
) -> T
where
    D: SolverDelegate<Interner = I>,
    I: Interner,
{
    // In case any fresh inference variables have been created between `state`
    // and the previous instantiation, extend `orig_values` for it.
    orig_values.extend(
        state.value.var_values.var_values.as_slice()[orig_values.len()..]
            .iter()
            .map(|&arg| delegate.fresh_var_for_kind_with_span(arg, span)),
    );

    let instantiation =
        EvalCtxt::compute_query_response_instantiation_values(delegate, orig_values, &state, span);

    let inspect::State { var_values, data } = delegate.instantiate_canonical(state, instantiation);

    EvalCtxt::unify_query_var_values(delegate, param_env, orig_values, var_values, span);
    data
}
