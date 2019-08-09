//! This module contains the code to instantiate a "query result", and
//! in particular to extract out the resulting region obligations and
//! encode them therein.
//!
//! For an overview of what canonicaliation is and how it fits into
//! rustc, check out the [chapter in the rustc guide][c].
//!
//! [c]: https://rust-lang.github.io/rustc-guide/traits/canonicalization.html

use crate::arena::ArenaAllocatable;
use crate::infer::canonical::substitute::substitute_value;
use crate::infer::canonical::{
    Canonical, CanonicalVarValues, CanonicalizedQueryResponse, Certainty,
    OriginalQueryValues, QueryRegionConstraints, QueryOutlivesConstraint, QueryResponse,
};
use crate::infer::region_constraints::{Constraint, RegionConstraintData};
use crate::infer::InferCtxtBuilder;
use crate::infer::{InferCtxt, InferOk, InferResult};
use crate::mir::interpret::ConstValue;
use rustc_data_structures::indexed_vec::Idx;
use rustc_data_structures::indexed_vec::IndexVec;
use std::fmt::Debug;
use syntax_pos::DUMMY_SP;
use crate::traits::query::{Fallible, NoSolution};
use crate::traits::TraitEngine;
use crate::traits::{Obligation, ObligationCause, PredicateObligation};
use crate::ty::fold::TypeFoldable;
use crate::ty::subst::{Kind, UnpackedKind};
use crate::ty::{self, BoundVar, InferConst, Ty, TyCtxt};
use crate::util::captures::Captures;

impl<'tcx> InferCtxtBuilder<'tcx> {
    /// The "main method" for a canonicalized trait query. Given the
    /// canonical key `canonical_key`, this method will create a new
    /// inference context, instantiate the key, and run your operation
    /// `op`. The operation should yield up a result (of type `R`) as
    /// well as a set of trait obligations that must be fully
    /// satisfied. These obligations will be processed and the
    /// canonical result created.
    ///
    /// Returns `NoSolution` in the event of any error.
    ///
    /// (It might be mildly nicer to implement this on `TyCtxt`, and
    /// not `InferCtxtBuilder`, but that is a bit tricky right now.
    /// In part because we would need a `for<'tcx>` sort of
    /// bound for the closure and in part because it is convenient to
    /// have `'tcx` be free on this function so that we can talk about
    /// `K: TypeFoldable<'tcx>`.)
    pub fn enter_canonical_trait_query<K, R>(
        &mut self,
        canonical_key: &Canonical<'tcx, K>,
        operation: impl FnOnce(&InferCtxt<'_, 'tcx>, &mut dyn TraitEngine<'tcx>, K) -> Fallible<R>,
    ) -> Fallible<CanonicalizedQueryResponse<'tcx, R>>
    where
        K: TypeFoldable<'tcx>,
        R: Debug + TypeFoldable<'tcx>,
        Canonical<'tcx, QueryResponse<'tcx, R>>: ArenaAllocatable,
    {
        self.enter_with_canonical(
            DUMMY_SP,
            canonical_key,
            |ref infcx, key, canonical_inference_vars| {
                let mut fulfill_cx = TraitEngine::new(infcx.tcx);
                let value = operation(infcx, &mut *fulfill_cx, key)?;
                infcx.make_canonicalized_query_response(
                    canonical_inference_vars,
                    value,
                    &mut *fulfill_cx
                )
            },
        )
    }
}

impl<'cx, 'tcx> InferCtxt<'cx, 'tcx> {
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
    pub fn make_canonicalized_query_response<T>(
        &self,
        inference_vars: CanonicalVarValues<'tcx>,
        answer: T,
        fulfill_cx: &mut dyn TraitEngine<'tcx>,
    ) -> Fallible<CanonicalizedQueryResponse<'tcx, T>>
    where
        T: Debug + TypeFoldable<'tcx>,
        Canonical<'tcx, QueryResponse<'tcx, T>>: ArenaAllocatable,
    {
        let query_response = self.make_query_response(inference_vars, answer, fulfill_cx)?;
        let canonical_result = self.canonicalize_response(&query_response);

        debug!(
            "make_canonicalized_query_response: canonical_result = {:#?}",
            canonical_result
        );

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
        self.canonicalize_response(&QueryResponse {
            var_values: inference_vars,
            region_constraints: QueryRegionConstraints::default(),
            certainty: Certainty::Proven, // Ambiguities are OK!
            value: answer,
        })
    }

    /// Helper for `make_canonicalized_query_response` that does
    /// everything up until the final canonicalization.
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

        debug!(
            "make_query_response(\
             inference_vars={:?}, \
             answer={:?})",
            inference_vars, answer,
        );

        // Select everything, returning errors.
        let true_errors = fulfill_cx.select_where_possible(self).err().unwrap_or_else(Vec::new);
        debug!("true_errors = {:#?}", true_errors);

        if !true_errors.is_empty() {
            // FIXME -- we don't indicate *why* we failed to solve
            debug!("make_query_response: true_errors={:#?}", true_errors);
            return Err(NoSolution);
        }

        // Anything left unselected *now* must be an ambiguity.
        let ambig_errors = fulfill_cx.select_all_or_error(self).err().unwrap_or_else(Vec::new);
        debug!("ambig_errors = {:#?}", ambig_errors);

        let region_obligations = self.take_registered_region_obligations();
        let region_constraints = self.with_region_constraints(|region_constraints| {
            make_query_region_constraints(
                tcx,
                region_obligations
                    .iter()
                    .map(|(_, r_o)| (r_o.sup_type, r_o.sub_region)),
                region_constraints,
            )
        });

        let certainty = if ambig_errors.is_empty() {
            Certainty::Proven
        } else {
            Certainty::Ambiguous
        };

        Ok(QueryResponse {
            var_values: inference_vars,
            region_constraints,
            certainty,
            value: answer,
        })
    }

    /// Given the (canonicalized) result to a canonical query,
    /// instantiates the result so it can be used, plugging in the
    /// values from the canonical query. (Note that the result may
    /// have been ambiguous; you should check the certainty level of
    /// the query before applying this function.)
    ///
    /// To get a good understanding of what is happening here, check
    /// out the [chapter in the rustc guide][c].
    ///
    /// [c]: https://rust-lang.github.io/rustc-guide/traits/canonicalization.html#processing-the-canonicalized-query-result
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
        let InferOk {
            value: result_subst,
            mut obligations,
        } = self.query_response_substitution(cause, param_env, original_values, query_response)?;

        obligations.extend(self.query_outlives_constraints_into_obligations(
            cause,
            param_env,
            &query_response.value.region_constraints.outlives,
            &result_subst,
        ));

        let user_result: R =
            query_response.substitute_projected(self.tcx, &result_subst, |q_r| &q_r.value);

        Ok(InferOk {
            value: user_result,
            obligations,
        })
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
        let result_subst =
            self.query_response_substitution_guess(cause, original_values, query_response);

        // Compute `QueryOutlivesConstraint` values that unify each of
        // the original values `v_o` that was canonicalized into a
        // variable...
        let mut obligations = vec![];

        for (index, original_value) in original_values.var_values.iter().enumerate() {
            // ...with the value `v_r` of that variable from the query.
            let result_value = query_response.substitute_projected(self.tcx, &result_subst, |v| {
                &v.var_values[BoundVar::new(index)]
            });
            match (original_value.unpack(), result_value.unpack()) {
                (UnpackedKind::Lifetime(ty::ReErased), UnpackedKind::Lifetime(ty::ReErased)) => {
                    // no action needed
                }

                (UnpackedKind::Lifetime(v_o), UnpackedKind::Lifetime(v_r)) => {
                    // To make `v_o = v_r`, we emit `v_o: v_r` and `v_r: v_o`.
                    if v_o != v_r {
                        output_query_region_constraints
                            .outlives
                            .push(ty::Binder::dummy(ty::OutlivesPredicate(v_o.into(), v_r)));
                        output_query_region_constraints
                            .outlives
                            .push(ty::Binder::dummy(ty::OutlivesPredicate(v_r.into(), v_o)));
                    }
                }

                (UnpackedKind::Type(v1), UnpackedKind::Type(v2)) => {
                    let ok = self.at(cause, param_env).eq(v1, v2)?;
                    obligations.extend(ok.into_obligations());
                }

                (UnpackedKind::Const(v1), UnpackedKind::Const(v2)) => {
                    let ok = self.at(cause, param_env).eq(v1, v2)?;
                    obligations.extend(ok.into_obligations());
                }

                _ => {
                    bug!(
                        "kind mismatch, cannot unify {:?} and {:?}",
                        original_value,
                        result_value
                    );
                }
            }
        }

        // ...also include the other query region constraints from the query.
        output_query_region_constraints.outlives.extend(
            query_response.value.region_constraints.outlives.iter().filter_map(|r_c| {
                let r_c = substitute_value(self.tcx, &result_subst, r_c);

                // Screen out `'a: 'a` cases -- we skip the binder here but
                // only compare the inner values to one another, so they are still at
                // consistent binding levels.
                let &ty::OutlivesPredicate(k1, r2) = r_c.skip_binder();
                if k1 != r2.into() {
                    Some(r_c)
                } else {
                    None
                }
            })
        );

        // ...also include the query member constraints.
        output_query_region_constraints.member_constraints.extend(
            query_response.value.region_constraints.member_constraints.iter().map(|p_c| {
                substitute_value(self.tcx, &result_subst, p_c)
            })
        );

        let user_result: R =
            query_response.substitute_projected(self.tcx, &result_subst, |q_r| &q_r.value);

        Ok(InferOk {
            value: user_result,
            obligations,
        })
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

        let result_subst =
            self.query_response_substitution_guess(cause, original_values, query_response);

        let obligations = self.unify_query_response_substitution_guess(
            cause,
            param_env,
            original_values,
            &result_subst,
            query_response,
        )?
            .into_obligations();

        Ok(InferOk {
            value: result_subst,
            obligations,
        })
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
        original_values: &OriginalQueryValues<'tcx>,
        query_response: &Canonical<'tcx, QueryResponse<'tcx, R>>,
    ) -> CanonicalVarValues<'tcx>
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
        assert!(universe_map.len() >= 1); // always have the root universe
        assert_eq!(
            universe_map[ty::UniverseIndex::ROOT.as_usize()],
            ty::UniverseIndex::ROOT
        );

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
        let mut opt_values: IndexVec<BoundVar, Option<Kind<'tcx>>> =
            IndexVec::from_elem_n(None, query_response.variables.len());

        // In terms of our example above, we are iterating over pairs like:
        // [(?A, Vec<?0>), ('static, '?1), (?B, ?0)]
        for (original_value, result_value) in original_values.var_values.iter().zip(result_values) {
            match result_value.unpack() {
                UnpackedKind::Type(result_value) => {
                    // e.g., here `result_value` might be `?0` in the example above...
                    if let ty::Bound(debruijn, b) = result_value.sty {
                        // ...in which case we would set `canonical_vars[0]` to `Some(?U)`.

                        // We only allow a `ty::INNERMOST` index in substitutions.
                        assert_eq!(debruijn, ty::INNERMOST);
                        opt_values[b.var] = Some(*original_value);
                    }
                }
                UnpackedKind::Lifetime(result_value) => {
                    // e.g., here `result_value` might be `'?1` in the example above...
                    if let &ty::RegionKind::ReLateBound(debruijn, br) = result_value {
                        // ... in which case we would set `canonical_vars[0]` to `Some('static)`.

                        // We only allow a `ty::INNERMOST` index in substitutions.
                        assert_eq!(debruijn, ty::INNERMOST);
                        opt_values[br.assert_bound_var()] = Some(*original_value);
                    }
                }
                UnpackedKind::Const(result_value) => {
                    if let ty::Const {
                        val: ConstValue::Infer(InferConst::Canonical(debrujin, b)),
                        ..
                    } = result_value {
                        // ...in which case we would set `canonical_vars[0]` to `Some(const X)`.

                        // We only allow a `ty::INNERMOST` index in substitutions.
                        assert_eq!(*debrujin, ty::INNERMOST);
                        opt_values[*b] = Some(*original_value);
                    }
                }
            }
        }

        // Create a result substitution: if we found a value for a
        // given variable in the loop above, use that. Otherwise, use
        // a fresh inference variable.
        let result_subst = CanonicalVarValues {
            var_values: query_response
                .variables
                .iter()
                .enumerate()
                .map(|(index, info)| {
                    if info.is_existential() {
                        match opt_values[BoundVar::new(index)] {
                            Some(k) => k,
                            None => self.instantiate_canonical_var(cause.span, *info, |u| {
                                universe_map[u.as_usize()]
                            }),
                        }
                    } else {
                        self.instantiate_canonical_var(cause.span, *info, |u| {
                            universe_map[u.as_usize()]
                        })
                    }
                })
                .collect(),
        };

        result_subst
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
        let substituted_query_response = |index: BoundVar| -> Kind<'tcx> {
            query_response.substitute_projected(self.tcx, &result_subst, |v| &v.var_values[index])
        };

        // Unify the original value for each variable with the value
        // taken from `query_response` (after applying `result_subst`).
        Ok(self.unify_canonical_vars(
            cause,
            param_env,
            original_values,
            substituted_query_response,
        )?)
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
        unsubstituted_region_constraints
            .iter()
            .map(move |constraint| {
                let constraint = substitute_value(self.tcx, result_subst, constraint);
                let &ty::OutlivesPredicate(k1, r2) = constraint.skip_binder(); // restored below

                Obligation::new(
                    cause.clone(),
                    param_env,
                    match k1.unpack() {
                        UnpackedKind::Lifetime(r1) => ty::Predicate::RegionOutlives(
                            ty::Binder::bind(
                                ty::OutlivesPredicate(r1, r2)
                            )
                        ),
                        UnpackedKind::Type(t1) => ty::Predicate::TypeOutlives(
                            ty::Binder::bind(
                                ty::OutlivesPredicate(t1, r2)
                            )
                        ),
                        UnpackedKind::Const(..) => {
                            // Consts cannot outlive one another, so we don't expect to
                            // ecounter this branch.
                            span_bug!(cause.span, "unexpected const outlives {:?}", constraint);
                        }
                    }
                )
            })
    }

    /// Given two sets of values for the same set of canonical variables, unify them.
    /// The second set is produced lazily by supplying indices from the first set.
    fn unify_canonical_vars(
        &self,
        cause: &ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        variables1: &OriginalQueryValues<'tcx>,
        variables2: impl Fn(BoundVar) -> Kind<'tcx>,
    ) -> InferResult<'tcx, ()> {
        self.commit_if_ok(|_| {
            let mut obligations = vec![];
            for (index, value1) in variables1.var_values.iter().enumerate() {
                let value2 = variables2(BoundVar::new(index));

                match (value1.unpack(), value2.unpack()) {
                    (UnpackedKind::Type(v1), UnpackedKind::Type(v2)) => {
                        obligations
                            .extend(self.at(cause, param_env).eq(v1, v2)?.into_obligations());
                    }
                    (
                        UnpackedKind::Lifetime(ty::ReErased),
                        UnpackedKind::Lifetime(ty::ReErased),
                    ) => {
                        // no action needed
                    }
                    (UnpackedKind::Lifetime(v1), UnpackedKind::Lifetime(v2)) => {
                        obligations
                            .extend(self.at(cause, param_env).eq(v1, v2)?.into_obligations());
                    }
                    (UnpackedKind::Const(v1), UnpackedKind::Const(v2)) => {
                        let ok = self.at(cause, param_env).eq(v1, v2)?;
                        obligations.extend(ok.into_obligations());
                    }
                    _ => {
                        bug!("kind mismatch, cannot unify {:?} and {:?}", value1, value2,);
                    }
                }
            }
            Ok(InferOk {
                value: (),
                obligations,
            })
        })
    }
}

/// Given the region obligations and constraints scraped from the infcx,
/// creates query region constraints.
pub fn make_query_region_constraints<'tcx>(
    tcx: TyCtxt<'tcx>,
    outlives_obligations: impl Iterator<Item = (Ty<'tcx>, ty::Region<'tcx>)>,
    region_constraints: &RegionConstraintData<'tcx>,
) -> QueryRegionConstraints<'tcx> {
    let RegionConstraintData {
        constraints,
        verifys,
        givens,
        member_constraints,
    } = region_constraints;

    assert!(verifys.is_empty());
    assert!(givens.is_empty());

    let outlives: Vec<_> = constraints
        .into_iter()
        .map(|(k, _)| match *k {
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
        })
        .map(ty::Binder::dummy) // no bound vars in the code above
        .chain(
            outlives_obligations
                .map(|(ty, r)| ty::OutlivesPredicate(ty.into(), r))
                .map(ty::Binder::dummy) // no bound vars in the code above
        )
        .collect();

    QueryRegionConstraints { outlives, member_constraints: member_constraints.clone() }
}
