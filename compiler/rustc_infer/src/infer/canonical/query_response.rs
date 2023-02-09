//! This module contains the code to instantiate a "query result", and
//! in particular to extract out the resulting region obligations and
//! encode them therein.
//!
//! For an overview of what canonicalization is and how it fits into
//! rustc, check out the [chapter in the rustc dev guide][c].
//!
//! [c]: https://rust-lang.github.io/chalk/book/canonical_queries/canonicalization.html

use crate::infer::canonical::substitute::{substitute_value, CanonicalExt};
use crate::infer::canonical::{
    Canonical, CanonicalQueryResponse, CanonicalVarValues, Certainty, OriginalQueryValues,
    QueryOutlivesConstraint, QueryRegionConstraints, QueryResponse,
};
use crate::infer::nll_relate::{NormalizationStrategy, TypeRelating, TypeRelatingDelegate};
use crate::infer::region_constraints::{Constraint, RegionConstraintData};
use crate::infer::{InferCtxt, InferOk, InferResult, NllRegionVariableOrigin};
use crate::traits::query::{Fallible, NoSolution};
use crate::traits::{Obligation, ObligationCause, PredicateObligation};
use crate::traits::{PredicateObligations, TraitEngine, TraitEngineExt};
use rustc_data_structures::captures::Captures;
use rustc_index::vec::Idx;
use rustc_index::vec::IndexVec;
use rustc_middle::arena::ArenaAllocatable;
use rustc_middle::mir::ConstraintCategory;
use rustc_middle::ty::fold::TypeFoldable;
use rustc_middle::ty::relate::TypeRelation;
use rustc_middle::ty::subst::{GenericArg, GenericArgKind};
use rustc_middle::ty::{self, BoundVar, ToPredicate, Ty, TyCtxt};
use rustc_span::Span;
use std::fmt::Debug;
use std::iter;

impl<'tcx> InferCtxt<'tcx> {
    /// This method is meant to be invoked as the final step of a canonical query
    /// implementation. It is given:
    ///
    /// - the instantiated variables `inference_vars` created from the query key
    /// - the result `answer` of the query
    /// - a fulfillment context `fulfill_cx` that may contain various obligations which
    ///   have yet to be proven.
    ///
    /// Given this, the function will process the obligations pending
    /// in `fulfill_cx`:
    ///
    /// - If all the obligations can be proven successfully, it will
    ///   package up any resulting region obligations (extracted from
    ///   `infcx`) along with the fully resolved value `answer` into a
    ///   query result (which is then itself canonicalized).
    /// - If some obligations can be neither proven nor disproven, then
    ///   the same thing happens, but the resulting query is marked as ambiguous.
    /// - Finally, if any of the obligations result in a hard error,
    ///   then `Err(NoSolution)` is returned.
    #[instrument(skip(self, inference_vars, answer, fulfill_cx), level = "trace")]
    pub fn make_canonicalized_query_response<T>(
        &self,
        inference_vars: CanonicalVarValues<'tcx>,
        answer: T,
        fulfill_cx: &mut dyn TraitEngine<'tcx>,
    ) -> Fallible<CanonicalQueryResponse<'tcx, T>>
    where
        T: Debug + TypeFoldable<'tcx>,
        Canonical<'tcx, QueryResponse<'tcx, T>>: ArenaAllocatable<'tcx>,
    {
        let query_response = self.make_query_response(inference_vars, answer, fulfill_cx)?;
        debug!("query_response = {:#?}", query_response);
        let canonical_result = self.canonicalize_response(query_response);
        debug!("canonical_result = {:#?}", canonical_result);

        Ok(self.tcx.arena.alloc(canonical_result))
    }

    /// A version of `make_canonicalized_query_response` that does
    /// not pack in obligations, for contexts that want to drop
    /// pending obligations instead of treating them as an ambiguity (e.g.
    /// typeck "probing" contexts).
    ///
    /// If you DO want to keep track of pending obligations (which
    /// include all region obligations, so this includes all cases
    /// that care about regions) with this function, you have to
    /// do it yourself, by e.g., having them be a part of the answer.
    pub fn make_query_response_ignoring_pending_obligations<T>(
        &self,
        inference_vars: CanonicalVarValues<'tcx>,
        answer: T,
    ) -> Canonical<'tcx, QueryResponse<'tcx, T>>
    where
        T: Debug + TypeFoldable<'tcx>,
    {
        self.canonicalize_response(QueryResponse {
            var_values: inference_vars,
            region_constraints: QueryRegionConstraints::default(),
            certainty: Certainty::Proven, // Ambiguities are OK!
            opaque_types: vec![],
            value: answer,
        })
    }

    /// Helper for `make_canonicalized_query_response` that does
    /// everything up until the final canonicalization.
    #[instrument(skip(self, fulfill_cx), level = "debug")]
    fn make_query_response<T>(
        &self,
        inference_vars: CanonicalVarValues<'tcx>,
        answer: T,
        fulfill_cx: &mut dyn TraitEngine<'tcx>,
    ) -> Result<QueryResponse<'tcx, T>, NoSolution>
    where
        T: Debug + TypeFoldable<'tcx>,
    {
        let tcx = self.tcx;

        // Select everything, returning errors.
        let true_errors = fulfill_cx.select_where_possible(self);
        debug!("true_errors = {:#?}", true_errors);

        if !true_errors.is_empty() {
            // FIXME -- we don't indicate *why* we failed to solve
            debug!("make_query_response: true_errors={:#?}", true_errors);
            return Err(NoSolution);
        }

        // Anything left unselected *now* must be an ambiguity.
        let ambig_errors = fulfill_cx.select_all_or_error(self);
        debug!("ambig_errors = {:#?}", ambig_errors);

        let region_obligations = self.take_registered_region_obligations();
        debug!(?region_obligations);
        let region_constraints = self.with_region_constraints(|region_constraints| {
            make_query_region_constraints(
                tcx,
                region_obligations
                    .iter()
                    .map(|r_o| (r_o.sup_type, r_o.sub_region, r_o.origin.to_constraint_category())),
                region_constraints,
            )
        });
        debug!(?region_constraints);

        let certainty =
            if ambig_errors.is_empty() { Certainty::Proven } else { Certainty::Ambiguous };

        let opaque_types = self.take_opaque_types_for_query_response();

        Ok(QueryResponse {
            var_values: inference_vars,
            region_constraints,
            certainty,
            value: answer,
            opaque_types,
        })
    }

    /// FIXME: This method should only be used for canonical queries and therefore be private.
    ///
    /// As the new solver does canonicalization slightly differently, this is also used there
    /// for now. This should hopefully change fairly soon.
    pub fn take_opaque_types_for_query_response(&self) -> Vec<(Ty<'tcx>, Ty<'tcx>)> {
        std::mem::take(&mut self.inner.borrow_mut().opaque_type_storage.opaque_types)
            .into_iter()
            .map(|(k, v)| (self.tcx.mk_opaque(k.def_id.to_def_id(), k.substs), v.hidden_type.ty))
            .collect()
    }

    /// Given the (canonicalized) result to a canonical query,
    /// instantiates the result so it can be used, plugging in the
    /// values from the canonical query. (Note that the result may
    /// have been ambiguous; you should check the certainty level of
    /// the query before applying this function.)
    ///
    /// To get a good understanding of what is happening here, check
    /// out the [chapter in the rustc dev guide][c].
    ///
    /// [c]: https://rust-lang.github.io/chalk/book/canonical_queries/canonicalization.html#processing-the-canonicalized-query-result
    pub fn instantiate_query_response_and_region_obligations<R>(
        &self,
        cause: &ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        original_values: &OriginalQueryValues<'tcx>,
        query_response: &Canonical<'tcx, QueryResponse<'tcx, R>>,
    ) -> InferResult<'tcx, R>
    where
        R: Debug + TypeFoldable<'tcx>,
    {
        let InferOk { value: result_subst, mut obligations } =
            self.query_response_substitution(cause, param_env, original_values, query_response)?;

        obligations.extend(self.query_outlives_constraints_into_obligations(
            cause,
            param_env,
            &query_response.value.region_constraints.outlives,
            &result_subst,
        ));

        let user_result: R =
            query_response.substitute_projected(self.tcx, &result_subst, |q_r| q_r.value.clone());

        Ok(InferOk { value: user_result, obligations })
    }

    /// An alternative to
    /// `instantiate_query_response_and_region_obligations` that is more
    /// efficient for NLL. NLL is a bit more advanced in the
    /// "transition to chalk" than the rest of the compiler. During
    /// the NLL type check, all of the "processing" of types and
    /// things happens in queries -- the NLL checker itself is only
    /// interested in the region obligations (`'a: 'b` or `T: 'b`)
    /// that come out of these queries, which it wants to convert into
    /// MIR-based constraints and solve. Therefore, it is most
    /// convenient for the NLL Type Checker to **directly consume**
    /// the `QueryOutlivesConstraint` values that arise from doing a
    /// query. This is contrast to other parts of the compiler, which
    /// would prefer for those `QueryOutlivesConstraint` to be converted
    /// into the older infcx-style constraints (e.g., calls to
    /// `sub_regions` or `register_region_obligation`).
    ///
    /// Therefore, `instantiate_nll_query_response_and_region_obligations` performs the same
    /// basic operations as `instantiate_query_response_and_region_obligations` but
    /// it returns its result differently:
    ///
    /// - It creates a substitution `S` that maps from the original
    ///   query variables to the values computed in the query
    ///   result. If any errors arise, they are propagated back as an
    ///   `Err` result.
    /// - In the case of a successful substitution, we will append
    ///   `QueryOutlivesConstraint` values onto the
    ///   `output_query_region_constraints` vector for the solver to
    ///   use (if an error arises, some values may also be pushed, but
    ///   they should be ignored).
    /// - It **can happen** (though it rarely does currently) that
    ///   equating types and things will give rise to subobligations
    ///   that must be processed. In this case, those subobligations
    ///   are propagated back in the return value.
    /// - Finally, the query result (of type `R`) is propagated back,
    ///   after applying the substitution `S`.
    pub fn instantiate_nll_query_response_and_region_obligations<R>(
        &self,
        cause: &ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        original_values: &OriginalQueryValues<'tcx>,
        query_response: &Canonical<'tcx, QueryResponse<'tcx, R>>,
        output_query_region_constraints: &mut QueryRegionConstraints<'tcx>,
    ) -> InferResult<'tcx, R>
    where
        R: Debug + TypeFoldable<'tcx>,
    {
        let InferOk { value: result_subst, mut obligations } = self
            .query_response_substitution_guess(cause, param_env, original_values, query_response)?;

        // Compute `QueryOutlivesConstraint` values that unify each of
        // the original values `v_o` that was canonicalized into a
        // variable...

        let constraint_category = cause.to_constraint_category();

        for (index, original_value) in original_values.var_values.iter().enumerate() {
            // ...with the value `v_r` of that variable from the query.
            let result_value = query_response.substitute_projected(self.tcx, &result_subst, |v| {
                v.var_values[BoundVar::new(index)]
            });
            match (original_value.unpack(), result_value.unpack()) {
                (GenericArgKind::Lifetime(re1), GenericArgKind::Lifetime(re2))
                    if re1.is_erased() && re2.is_erased() =>
                {
                    // No action needed.
                }

                (GenericArgKind::Lifetime(v_o), GenericArgKind::Lifetime(v_r)) => {
                    // To make `v_o = v_r`, we emit `v_o: v_r` and `v_r: v_o`.
                    if v_o != v_r {
                        output_query_region_constraints
                            .outlives
                            .push((ty::OutlivesPredicate(v_o.into(), v_r), constraint_category));
                        output_query_region_constraints
                            .outlives
                            .push((ty::OutlivesPredicate(v_r.into(), v_o), constraint_category));
                    }
                }

                (GenericArgKind::Type(v1), GenericArgKind::Type(v2)) => {
                    TypeRelating::new(
                        self,
                        QueryTypeRelatingDelegate {
                            infcx: self,
                            param_env,
                            cause,
                            obligations: &mut obligations,
                        },
                        ty::Variance::Invariant,
                    )
                    .relate(v1, v2)?;
                }

                (GenericArgKind::Const(v1), GenericArgKind::Const(v2)) => {
                    TypeRelating::new(
                        self,
                        QueryTypeRelatingDelegate {
                            infcx: self,
                            param_env,
                            cause,
                            obligations: &mut obligations,
                        },
                        ty::Variance::Invariant,
                    )
                    .relate(v1, v2)?;
                }

                _ => {
                    bug!("kind mismatch, cannot unify {:?} and {:?}", original_value, result_value);
                }
            }
        }

        // ...also include the other query region constraints from the query.
        output_query_region_constraints.outlives.extend(
            query_response.value.region_constraints.outlives.iter().filter_map(|&r_c| {
                let r_c = substitute_value(self.tcx, &result_subst, r_c);

                // Screen out `'a: 'a` cases.
                let ty::OutlivesPredicate(k1, r2) = r_c.0;
                if k1 != r2.into() { Some(r_c) } else { None }
            }),
        );

        // ...also include the query member constraints.
        output_query_region_constraints.member_constraints.extend(
            query_response
                .value
                .region_constraints
                .member_constraints
                .iter()
                .map(|p_c| substitute_value(self.tcx, &result_subst, p_c.clone())),
        );

        let user_result: R =
            query_response.substitute_projected(self.tcx, &result_subst, |q_r| q_r.value.clone());

        Ok(InferOk { value: user_result, obligations })
    }

    /// Given the original values and the (canonicalized) result from
    /// computing a query, returns a substitution that can be applied
    /// to the query result to convert the result back into the
    /// original namespace.
    ///
    /// The substitution also comes accompanied with subobligations
    /// that arose from unification; these might occur if (for
    /// example) we are doing lazy normalization and the value
    /// assigned to a type variable is unified with an unnormalized
    /// projection.
    fn query_response_substitution<R>(
        &self,
        cause: &ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        original_values: &OriginalQueryValues<'tcx>,
        query_response: &Canonical<'tcx, QueryResponse<'tcx, R>>,
    ) -> InferResult<'tcx, CanonicalVarValues<'tcx>>
    where
        R: Debug + TypeFoldable<'tcx>,
    {
        debug!(
            "query_response_substitution(original_values={:#?}, query_response={:#?})",
            original_values, query_response,
        );

        let mut value = self.query_response_substitution_guess(
            cause,
            param_env,
            original_values,
            query_response,
        )?;

        value.obligations.extend(
            self.unify_query_response_substitution_guess(
                cause,
                param_env,
                original_values,
                &value.value,
                query_response,
            )?
            .into_obligations(),
        );

        Ok(value)
    }

    /// Given the original values and the (canonicalized) result from
    /// computing a query, returns a **guess** at a substitution that
    /// can be applied to the query result to convert the result back
    /// into the original namespace. This is called a **guess**
    /// because it uses a quick heuristic to find the values for each
    /// canonical variable; if that quick heuristic fails, then we
    /// will instantiate fresh inference variables for each canonical
    /// variable instead. Therefore, the result of this method must be
    /// properly unified
    fn query_response_substitution_guess<R>(
        &self,
        cause: &ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        original_values: &OriginalQueryValues<'tcx>,
        query_response: &Canonical<'tcx, QueryResponse<'tcx, R>>,
    ) -> InferResult<'tcx, CanonicalVarValues<'tcx>>
    where
        R: Debug + TypeFoldable<'tcx>,
    {
        debug!(
            "query_response_substitution_guess(original_values={:#?}, query_response={:#?})",
            original_values, query_response,
        );

        // For each new universe created in the query result that did
        // not appear in the original query, create a local
        // superuniverse.
        let mut universe_map = original_values.universe_map.clone();
        let num_universes_in_query = original_values.universe_map.len();
        let num_universes_in_response = query_response.max_universe.as_usize() + 1;
        for _ in num_universes_in_query..num_universes_in_response {
            universe_map.push(self.create_next_universe());
        }
        assert!(!universe_map.is_empty()); // always have the root universe
        assert_eq!(universe_map[ty::UniverseIndex::ROOT.as_usize()], ty::UniverseIndex::ROOT);

        // Every canonical query result includes values for each of
        // the inputs to the query. Therefore, we begin by unifying
        // these values with the original inputs that were
        // canonicalized.
        let result_values = &query_response.value.var_values;
        assert_eq!(original_values.var_values.len(), result_values.len());

        // Quickly try to find initial values for the canonical
        // variables in the result in terms of the query. We do this
        // by iterating down the values that the query gave to each of
        // the canonical inputs. If we find that one of those values
        // is directly equal to one of the canonical variables in the
        // result, then we can type the corresponding value from the
        // input. See the example above.
        let mut opt_values: IndexVec<BoundVar, Option<GenericArg<'tcx>>> =
            IndexVec::from_elem_n(None, query_response.variables.len());

        // In terms of our example above, we are iterating over pairs like:
        // [(?A, Vec<?0>), ('static, '?1), (?B, ?0)]
        for (original_value, result_value) in iter::zip(&original_values.var_values, result_values)
        {
            match result_value.unpack() {
                GenericArgKind::Type(result_value) => {
                    // e.g., here `result_value` might be `?0` in the example above...
                    if let ty::Bound(debruijn, b) = *result_value.kind() {
                        // ...in which case we would set `canonical_vars[0]` to `Some(?U)`.

                        // We only allow a `ty::INNERMOST` index in substitutions.
                        assert_eq!(debruijn, ty::INNERMOST);
                        opt_values[b.var] = Some(*original_value);
                    }
                }
                GenericArgKind::Lifetime(result_value) => {
                    // e.g., here `result_value` might be `'?1` in the example above...
                    if let ty::ReLateBound(debruijn, br) = *result_value {
                        // ... in which case we would set `canonical_vars[0]` to `Some('static)`.

                        // We only allow a `ty::INNERMOST` index in substitutions.
                        assert_eq!(debruijn, ty::INNERMOST);
                        opt_values[br.var] = Some(*original_value);
                    }
                }
                GenericArgKind::Const(result_value) => {
                    if let ty::ConstKind::Bound(debrujin, b) = result_value.kind() {
                        // ...in which case we would set `canonical_vars[0]` to `Some(const X)`.

                        // We only allow a `ty::INNERMOST` index in substitutions.
                        assert_eq!(debrujin, ty::INNERMOST);
                        opt_values[b] = Some(*original_value);
                    }
                }
            }
        }

        // Create a result substitution: if we found a value for a
        // given variable in the loop above, use that. Otherwise, use
        // a fresh inference variable.
        let result_subst = CanonicalVarValues {
            var_values: self.tcx.mk_substs(query_response.variables.iter().enumerate().map(
                |(index, info)| {
                    if info.is_existential() {
                        match opt_values[BoundVar::new(index)] {
                            Some(k) => k,
                            None => self.instantiate_canonical_var(cause.span, info, |u| {
                                universe_map[u.as_usize()]
                            }),
                        }
                    } else {
                        self.instantiate_canonical_var(cause.span, info, |u| {
                            universe_map[u.as_usize()]
                        })
                    }
                },
            )),
        };

        let mut obligations = vec![];

        // Carry all newly resolved opaque types to the caller's scope
        for &(a, b) in &query_response.value.opaque_types {
            let a = substitute_value(self.tcx, &result_subst, a);
            let b = substitute_value(self.tcx, &result_subst, b);
            obligations.extend(self.at(cause, param_env).eq(a, b)?.obligations);
        }

        Ok(InferOk { value: result_subst, obligations })
    }

    /// Given a "guess" at the values for the canonical variables in
    /// the input, try to unify with the *actual* values found in the
    /// query result. Often, but not always, this is a no-op, because
    /// we already found the mapping in the "guessing" step.
    ///
    /// See also: `query_response_substitution_guess`
    fn unify_query_response_substitution_guess<R>(
        &self,
        cause: &ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        original_values: &OriginalQueryValues<'tcx>,
        result_subst: &CanonicalVarValues<'tcx>,
        query_response: &Canonical<'tcx, QueryResponse<'tcx, R>>,
    ) -> InferResult<'tcx, ()>
    where
        R: Debug + TypeFoldable<'tcx>,
    {
        // A closure that yields the result value for the given
        // canonical variable; this is taken from
        // `query_response.var_values` after applying the substitution
        // `result_subst`.
        let substituted_query_response = |index: BoundVar| -> GenericArg<'tcx> {
            query_response.substitute_projected(self.tcx, &result_subst, |v| v.var_values[index])
        };

        // Unify the original value for each variable with the value
        // taken from `query_response` (after applying `result_subst`).
        self.unify_canonical_vars(cause, param_env, original_values, substituted_query_response)
    }

    /// Converts the region constraints resulting from a query into an
    /// iterator of obligations.
    fn query_outlives_constraints_into_obligations<'a>(
        &'a self,
        cause: &'a ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        unsubstituted_region_constraints: &'a [QueryOutlivesConstraint<'tcx>],
        result_subst: &'a CanonicalVarValues<'tcx>,
    ) -> impl Iterator<Item = PredicateObligation<'tcx>> + 'a + Captures<'tcx> {
        unsubstituted_region_constraints.iter().map(move |&constraint| {
            let predicate = substitute_value(self.tcx, result_subst, constraint);
            self.query_outlives_constraint_to_obligation(predicate, cause.clone(), param_env)
        })
    }

    pub fn query_outlives_constraint_to_obligation(
        &self,
        (predicate, _): QueryOutlivesConstraint<'tcx>,
        cause: ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
    ) -> Obligation<'tcx, ty::Predicate<'tcx>> {
        let ty::OutlivesPredicate(k1, r2) = predicate;

        let atom = match k1.unpack() {
            GenericArgKind::Lifetime(r1) => {
                ty::PredicateKind::Clause(ty::Clause::RegionOutlives(ty::OutlivesPredicate(r1, r2)))
            }
            GenericArgKind::Type(t1) => {
                ty::PredicateKind::Clause(ty::Clause::TypeOutlives(ty::OutlivesPredicate(t1, r2)))
            }
            GenericArgKind::Const(..) => {
                // Consts cannot outlive one another, so we don't expect to
                // encounter this branch.
                span_bug!(cause.span, "unexpected const outlives {:?}", predicate);
            }
        };
        let predicate = ty::Binder::dummy(atom);

        Obligation::new(self.tcx, cause, param_env, predicate)
    }

    /// Given two sets of values for the same set of canonical variables, unify them.
    /// The second set is produced lazily by supplying indices from the first set.
    fn unify_canonical_vars(
        &self,
        cause: &ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        variables1: &OriginalQueryValues<'tcx>,
        variables2: impl Fn(BoundVar) -> GenericArg<'tcx>,
    ) -> InferResult<'tcx, ()> {
        self.commit_if_ok(|_| {
            let mut obligations = vec![];
            for (index, value1) in variables1.var_values.iter().enumerate() {
                let value2 = variables2(BoundVar::new(index));

                match (value1.unpack(), value2.unpack()) {
                    (GenericArgKind::Type(v1), GenericArgKind::Type(v2)) => {
                        obligations
                            .extend(self.at(cause, param_env).eq(v1, v2)?.into_obligations());
                    }
                    (GenericArgKind::Lifetime(re1), GenericArgKind::Lifetime(re2))
                        if re1.is_erased() && re2.is_erased() =>
                    {
                        // no action needed
                    }
                    (GenericArgKind::Lifetime(v1), GenericArgKind::Lifetime(v2)) => {
                        obligations
                            .extend(self.at(cause, param_env).eq(v1, v2)?.into_obligations());
                    }
                    (GenericArgKind::Const(v1), GenericArgKind::Const(v2)) => {
                        let ok = self.at(cause, param_env).eq(v1, v2)?;
                        obligations.extend(ok.into_obligations());
                    }
                    _ => {
                        bug!("kind mismatch, cannot unify {:?} and {:?}", value1, value2,);
                    }
                }
            }
            Ok(InferOk { value: (), obligations })
        })
    }
}

/// Given the region obligations and constraints scraped from the infcx,
/// creates query region constraints.
pub fn make_query_region_constraints<'tcx>(
    tcx: TyCtxt<'tcx>,
    outlives_obligations: impl Iterator<Item = (Ty<'tcx>, ty::Region<'tcx>, ConstraintCategory<'tcx>)>,
    region_constraints: &RegionConstraintData<'tcx>,
) -> QueryRegionConstraints<'tcx> {
    let RegionConstraintData { constraints, verifys, givens, member_constraints } =
        region_constraints;

    assert!(verifys.is_empty());
    assert!(givens.is_empty());

    debug!(?constraints);

    let outlives: Vec<_> = constraints
        .iter()
        .map(|(k, origin)| {
            let constraint = match *k {
                // Swap regions because we are going from sub (<=) to outlives
                // (>=).
                Constraint::VarSubVar(v1, v2) => ty::OutlivesPredicate(
                    tcx.mk_region(ty::ReVar(v2)).into(),
                    tcx.mk_region(ty::ReVar(v1)),
                ),
                Constraint::VarSubReg(v1, r2) => {
                    ty::OutlivesPredicate(r2.into(), tcx.mk_region(ty::ReVar(v1)))
                }
                Constraint::RegSubVar(r1, v2) => {
                    ty::OutlivesPredicate(tcx.mk_region(ty::ReVar(v2)).into(), r1)
                }
                Constraint::RegSubReg(r1, r2) => ty::OutlivesPredicate(r2.into(), r1),
            };
            (constraint, origin.to_constraint_category())
        })
        .chain(outlives_obligations.map(|(ty, r, constraint_category)| {
            (ty::OutlivesPredicate(ty.into(), r), constraint_category)
        }))
        .collect();

    QueryRegionConstraints { outlives, member_constraints: member_constraints.clone() }
}

struct QueryTypeRelatingDelegate<'a, 'tcx> {
    infcx: &'a InferCtxt<'tcx>,
    obligations: &'a mut Vec<PredicateObligation<'tcx>>,
    param_env: ty::ParamEnv<'tcx>,
    cause: &'a ObligationCause<'tcx>,
}

impl<'tcx> TypeRelatingDelegate<'tcx> for QueryTypeRelatingDelegate<'_, 'tcx> {
    fn span(&self) -> Span {
        self.cause.span
    }

    fn param_env(&self) -> ty::ParamEnv<'tcx> {
        self.param_env
    }

    fn create_next_universe(&mut self) -> ty::UniverseIndex {
        self.infcx.create_next_universe()
    }

    fn next_existential_region_var(&mut self, from_forall: bool) -> ty::Region<'tcx> {
        let origin = NllRegionVariableOrigin::Existential { from_forall };
        self.infcx.next_nll_region_var(origin)
    }

    fn next_placeholder_region(&mut self, placeholder: ty::PlaceholderRegion) -> ty::Region<'tcx> {
        self.infcx.tcx.mk_region(ty::RePlaceholder(placeholder))
    }

    fn generalize_existential(&mut self, universe: ty::UniverseIndex) -> ty::Region<'tcx> {
        self.infcx.next_nll_region_var_in_universe(
            NllRegionVariableOrigin::Existential { from_forall: false },
            universe,
        )
    }

    fn push_outlives(
        &mut self,
        sup: ty::Region<'tcx>,
        sub: ty::Region<'tcx>,
        _info: ty::VarianceDiagInfo<'tcx>,
    ) {
        self.obligations.push(Obligation {
            cause: self.cause.clone(),
            param_env: self.param_env,
            predicate: ty::Binder::dummy(ty::PredicateKind::Clause(ty::Clause::RegionOutlives(
                ty::OutlivesPredicate(sup, sub),
            )))
            .to_predicate(self.infcx.tcx),
            recursion_depth: 0,
        });
    }

    fn normalization() -> NormalizationStrategy {
        NormalizationStrategy::Eager
    }

    fn forbid_inference_vars() -> bool {
        true
    }

    fn register_obligations(&mut self, obligations: PredicateObligations<'tcx>) {
        self.obligations.extend(obligations);
    }
}
