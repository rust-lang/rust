//! This module contains code to instantiate new values into a
//! `Canonical<'db, T>`.
//!
//! For an overview of what canonicalization is and how it fits into
//! rustc, check out the [chapter in the rustc dev guide][c].
//!
//! [c]: https://rust-lang.github.io/chalk/book/canonical_queries/canonicalization.html

use std::{fmt::Debug, iter};

use crate::next_solver::{
    BoundConst, BoundRegion, BoundTy, Canonical, CanonicalVarKind, CanonicalVarValues, Clauses,
    Const, ConstKind, DbInterner, GenericArg, ParamEnv, Predicate, Region, RegionKind, Ty, TyKind,
    fold::FnMutDelegate,
    infer::{
        InferCtxt, InferOk, InferResult,
        canonical::{QueryRegionConstraints, QueryResponse, canonicalizer::OriginalQueryValues},
        opaque_types::table::OpaqueTypeStorageEntries,
        traits::{ObligationCause, PredicateObligations},
    },
};
use rustc_hash::FxHashMap;
use rustc_index::{Idx as _, IndexVec};
use rustc_type_ir::{
    BoundVar, BoundVarIndexKind, GenericArgKind, TypeFlags, TypeFoldable, TypeFolder,
    TypeSuperFoldable, TypeVisitableExt, UniverseIndex,
    inherent::{GenericArg as _, IntoKind},
};
use tracing::{debug, instrument};

pub trait CanonicalExt<'db, V> {
    fn instantiate(&self, tcx: DbInterner<'db>, var_values: &CanonicalVarValues<'db>) -> V
    where
        V: TypeFoldable<DbInterner<'db>>;
    fn instantiate_projected<T>(
        &self,
        tcx: DbInterner<'db>,
        var_values: &CanonicalVarValues<'db>,
        projection_fn: impl FnOnce(&V) -> T,
    ) -> T
    where
        T: TypeFoldable<DbInterner<'db>>;
}

/// FIXME(-Znext-solver): This or public because it is shared with the
/// new trait solver implementation. We should deduplicate canonicalization.
impl<'db, V> CanonicalExt<'db, V> for Canonical<'db, V> {
    /// Instantiate the wrapped value, replacing each canonical value
    /// with the value given in `var_values`.
    fn instantiate(&self, tcx: DbInterner<'db>, var_values: &CanonicalVarValues<'db>) -> V
    where
        V: TypeFoldable<DbInterner<'db>>,
    {
        self.instantiate_projected(tcx, var_values, |value| value.clone())
    }

    /// Allows one to apply a instantiation to some subset of
    /// `self.value`. Invoke `projection_fn` with `self.value` to get
    /// a value V that is expressed in terms of the same canonical
    /// variables bound in `self` (usually this extracts from subset
    /// of `self`). Apply the instantiation `var_values` to this value
    /// V, replacing each of the canonical variables.
    fn instantiate_projected<T>(
        &self,
        tcx: DbInterner<'db>,
        var_values: &CanonicalVarValues<'db>,
        projection_fn: impl FnOnce(&V) -> T,
    ) -> T
    where
        T: TypeFoldable<DbInterner<'db>>,
    {
        assert_eq!(self.variables.len(), var_values.len());
        let value = projection_fn(&self.value);
        instantiate_value(tcx, var_values, value)
    }
}

/// Instantiate the values from `var_values` into `value`. `var_values`
/// must be values for the set of canonical variables that appear in
/// `value`.
pub(super) fn instantiate_value<'db, T>(
    tcx: DbInterner<'db>,
    var_values: &CanonicalVarValues<'db>,
    value: T,
) -> T
where
    T: TypeFoldable<DbInterner<'db>>,
{
    if var_values.var_values.is_empty() {
        value
    } else {
        let delegate = FnMutDelegate {
            regions: &mut |br: BoundRegion| match var_values[br.var].kind() {
                GenericArgKind::Lifetime(l) => l,
                r => panic!("{br:?} is a region but value is {r:?}"),
            },
            types: &mut |bound_ty: BoundTy| match var_values[bound_ty.var].kind() {
                GenericArgKind::Type(ty) => ty,
                r => panic!("{bound_ty:?} is a type but value is {r:?}"),
            },
            consts: &mut |bound_ct: BoundConst| match var_values[bound_ct.var].kind() {
                GenericArgKind::Const(ct) => ct,
                c => panic!("{bound_ct:?} is a const but value is {c:?}"),
            },
        };

        let value = tcx.replace_escaping_bound_vars_uncached(value, delegate);
        value.fold_with(&mut CanonicalInstantiator {
            tcx,
            var_values: var_values.var_values.as_slice(),
            cache: Default::default(),
        })
    }
}

/// Replaces the bound vars in a canonical binder with var values.
struct CanonicalInstantiator<'db, 'a> {
    tcx: DbInterner<'db>,

    // The values that the bound vars are being instantiated with.
    var_values: &'a [GenericArg<'db>],

    // Because we use `BoundVarIndexKind::Canonical`, we can cache
    // based only on the entire ty, not worrying about a `DebruijnIndex`
    cache: FxHashMap<Ty<'db>, Ty<'db>>,
}

impl<'db, 'a> TypeFolder<DbInterner<'db>> for CanonicalInstantiator<'db, 'a> {
    fn cx(&self) -> DbInterner<'db> {
        self.tcx
    }

    fn fold_ty(&mut self, t: Ty<'db>) -> Ty<'db> {
        match t.kind() {
            TyKind::Bound(BoundVarIndexKind::Canonical, bound_ty) => {
                self.var_values[bound_ty.var.as_usize()].expect_ty()
            }
            _ => {
                if !t.has_type_flags(TypeFlags::HAS_CANONICAL_BOUND) {
                    t
                } else if let Some(&t) = self.cache.get(&t) {
                    t
                } else {
                    let res = t.super_fold_with(self);
                    assert!(self.cache.insert(t, res).is_none());
                    res
                }
            }
        }
    }

    fn fold_region(&mut self, r: Region<'db>) -> Region<'db> {
        match r.kind() {
            RegionKind::ReBound(BoundVarIndexKind::Canonical, br) => {
                self.var_values[br.var.as_usize()].expect_region()
            }
            _ => r,
        }
    }

    fn fold_const(&mut self, ct: Const<'db>) -> Const<'db> {
        match ct.kind() {
            ConstKind::Bound(BoundVarIndexKind::Canonical, bound_const) => {
                self.var_values[bound_const.var.as_usize()].expect_const()
            }
            _ => ct.super_fold_with(self),
        }
    }

    fn fold_predicate(&mut self, p: Predicate<'db>) -> Predicate<'db> {
        if p.has_type_flags(TypeFlags::HAS_CANONICAL_BOUND) { p.super_fold_with(self) } else { p }
    }

    fn fold_clauses(&mut self, c: Clauses<'db>) -> Clauses<'db> {
        if !c.has_type_flags(TypeFlags::HAS_CANONICAL_BOUND) {
            return c;
        }

        // FIXME: We might need cache here for perf like rustc
        c.super_fold_with(self)
    }
}

impl<'db> InferCtxt<'db> {
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
        inference_vars: CanonicalVarValues<'db>,
        answer: T,
        prev_entries: OpaqueTypeStorageEntries,
    ) -> Canonical<'db, QueryResponse<'db, T>>
    where
        T: TypeFoldable<DbInterner<'db>>,
    {
        // While we ignore region constraints and pending obligations,
        // we do return constrained opaque types to avoid unconstrained
        // inference variables in the response. This is important as we want
        // to check that opaques in deref steps stay unconstrained.
        //
        // This doesn't handle the more general case for non-opaques as
        // ambiguous `Projection` obligations have same the issue.
        let opaque_types = self
            .inner
            .borrow_mut()
            .opaque_type_storage
            .opaque_types_added_since(prev_entries)
            .map(|(k, v)| (k, v.ty))
            .collect();

        self.canonicalize_response(QueryResponse {
            var_values: inference_vars,
            region_constraints: QueryRegionConstraints::default(),
            opaque_types,
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
    /// out the [chapter in the rustc dev guide][c].
    ///
    /// [c]: https://rust-lang.github.io/chalk/book/canonical_queries/canonicalization.html#processing-the-canonicalized-query-result
    pub fn instantiate_query_response_and_region_obligations<R>(
        &self,
        cause: &ObligationCause,
        param_env: ParamEnv<'db>,
        original_values: &OriginalQueryValues<'db>,
        query_response: &Canonical<'db, QueryResponse<'db, R>>,
    ) -> InferResult<'db, R>
    where
        R: TypeFoldable<DbInterner<'db>>,
    {
        let InferOk { value: result_args, obligations } =
            self.query_response_instantiation(cause, param_env, original_values, query_response)?;

        for predicate in &query_response.value.region_constraints.outlives {
            let predicate = instantiate_value(self.interner, &result_args, *predicate);
            self.register_outlives_constraint(predicate);
        }

        for assumption in &query_response.value.region_constraints.assumptions {
            let assumption = instantiate_value(self.interner, &result_args, *assumption);
            self.register_region_assumption(assumption);
        }

        let user_result: R =
            query_response
                .instantiate_projected(self.interner, &result_args, |q_r| q_r.value.clone());

        Ok(InferOk { value: user_result, obligations })
    }

    /// Given the original values and the (canonicalized) result from
    /// computing a query, returns an instantiation that can be applied
    /// to the query result to convert the result back into the
    /// original namespace.
    ///
    /// The instantiation also comes accompanied with subobligations
    /// that arose from unification; these might occur if (for
    /// example) we are doing lazy normalization and the value
    /// assigned to a type variable is unified with an unnormalized
    /// projection.
    fn query_response_instantiation<R>(
        &self,
        cause: &ObligationCause,
        param_env: ParamEnv<'db>,
        original_values: &OriginalQueryValues<'db>,
        query_response: &Canonical<'db, QueryResponse<'db, R>>,
    ) -> InferResult<'db, CanonicalVarValues<'db>>
    where
        R: Debug + TypeFoldable<DbInterner<'db>>,
    {
        debug!(
            "query_response_instantiation(original_values={:#?}, query_response={:#?})",
            original_values, query_response,
        );

        let mut value = self.query_response_instantiation_guess(
            cause,
            param_env,
            original_values,
            query_response,
        )?;

        value.obligations.extend(
            self.unify_query_response_instantiation_guess(
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
    /// computing a query, returns a **guess** at an instantiation that
    /// can be applied to the query result to convert the result back
    /// into the original namespace. This is called a **guess**
    /// because it uses a quick heuristic to find the values for each
    /// canonical variable; if that quick heuristic fails, then we
    /// will instantiate fresh inference variables for each canonical
    /// variable instead. Therefore, the result of this method must be
    /// properly unified
    #[instrument(level = "debug", skip(self, param_env))]
    fn query_response_instantiation_guess<R>(
        &self,
        cause: &ObligationCause,
        param_env: ParamEnv<'db>,
        original_values: &OriginalQueryValues<'db>,
        query_response: &Canonical<'db, QueryResponse<'db, R>>,
    ) -> InferResult<'db, CanonicalVarValues<'db>>
    where
        R: Debug + TypeFoldable<DbInterner<'db>>,
    {
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
        assert_eq!(universe_map[UniverseIndex::ROOT.as_usize()], UniverseIndex::ROOT);

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
        let mut opt_values: IndexVec<BoundVar, Option<GenericArg<'db>>> =
            IndexVec::from_elem_n(None, query_response.variables.len());

        for (original_value, result_value) in iter::zip(&original_values.var_values, result_values)
        {
            match result_value.kind() {
                GenericArgKind::Type(result_value) => {
                    // We disable the instantiation guess for inference variables
                    // and only use it for placeholders. We need to handle the
                    // `sub_root` of type inference variables which would make this
                    // more involved. They are also a lot rarer than region variables.
                    if let TyKind::Bound(index_kind, b) = result_value.kind()
                        && !matches!(
                            query_response.variables.as_slice()[b.var.as_usize()],
                            CanonicalVarKind::Ty { .. }
                        )
                    {
                        // We only allow a `Canonical` index in generic parameters.
                        assert!(matches!(index_kind, BoundVarIndexKind::Canonical));
                        opt_values[b.var] = Some(*original_value);
                    }
                }
                GenericArgKind::Lifetime(result_value) => {
                    if let RegionKind::ReBound(index_kind, b) = result_value.kind() {
                        // We only allow a `Canonical` index in generic parameters.
                        assert!(matches!(index_kind, BoundVarIndexKind::Canonical));
                        opt_values[b.var] = Some(*original_value);
                    }
                }
                GenericArgKind::Const(result_value) => {
                    if let ConstKind::Bound(index_kind, b) = result_value.kind() {
                        // We only allow a `Canonical` index in generic parameters.
                        assert!(matches!(index_kind, BoundVarIndexKind::Canonical));
                        opt_values[b.var] = Some(*original_value);
                    }
                }
            }
        }

        // Create result arguments: if we found a value for a
        // given variable in the loop above, use that. Otherwise, use
        // a fresh inference variable.
        let interner = self.interner;
        let variables = query_response.variables;
        let var_values =
            CanonicalVarValues::instantiate(interner, variables, |var_values, kind| {
                if kind.universe() != UniverseIndex::ROOT {
                    // A variable from inside a binder of the query. While ideally these shouldn't
                    // exist at all, we have to deal with them for now.
                    self.instantiate_canonical_var(kind, var_values, |u| universe_map[u.as_usize()])
                } else if kind.is_existential() {
                    match opt_values[BoundVar::new(var_values.len())] {
                        Some(k) => k,
                        None => self.instantiate_canonical_var(kind, var_values, |u| {
                            universe_map[u.as_usize()]
                        }),
                    }
                } else {
                    // For placeholders which were already part of the input, we simply map this
                    // universal bound variable back the placeholder of the input.
                    opt_values[BoundVar::new(var_values.len())]
                        .expect("expected placeholder to be unified with itself during response")
                }
            });

        let mut obligations = PredicateObligations::new();

        // Carry all newly resolved opaque types to the caller's scope
        for &(a, b) in &query_response.value.opaque_types {
            let a = instantiate_value(self.interner, &var_values, a);
            let b = instantiate_value(self.interner, &var_values, b);
            debug!(?a, ?b, "constrain opaque type");
            // We use equate here instead of, for example, just registering the
            // opaque type's hidden value directly, because the hidden type may have been an inference
            // variable that got constrained to the opaque type itself. In that case we want to equate
            // the generic args of the opaque with the generic params of its hidden type version.
            obligations.extend(
                self.at(cause, param_env)
                    .eq(Ty::new_opaque(self.interner, a.def_id, a.args), b)?
                    .obligations,
            );
        }

        Ok(InferOk { value: var_values, obligations })
    }

    /// Given a "guess" at the values for the canonical variables in
    /// the input, try to unify with the *actual* values found in the
    /// query result. Often, but not always, this is a no-op, because
    /// we already found the mapping in the "guessing" step.
    ///
    /// See also: [`Self::query_response_instantiation_guess`]
    fn unify_query_response_instantiation_guess<R>(
        &self,
        cause: &ObligationCause,
        param_env: ParamEnv<'db>,
        original_values: &OriginalQueryValues<'db>,
        result_args: &CanonicalVarValues<'db>,
        query_response: &Canonical<'db, QueryResponse<'db, R>>,
    ) -> InferResult<'db, ()>
    where
        R: Debug + TypeFoldable<DbInterner<'db>>,
    {
        // A closure that yields the result value for the given
        // canonical variable; this is taken from
        // `query_response.var_values` after applying the instantiation
        // by `result_args`.
        let instantiated_query_response = |index: BoundVar| -> GenericArg<'db> {
            query_response
                .instantiate_projected(self.interner, result_args, |v| v.var_values[index])
        };

        // Unify the original value for each variable with the value
        // taken from `query_response` (after applying `result_args`).
        self.unify_canonical_vars(cause, param_env, original_values, instantiated_query_response)
    }

    /// Given two sets of values for the same set of canonical variables, unify them.
    /// The second set is produced lazily by supplying indices from the first set.
    fn unify_canonical_vars(
        &self,
        cause: &ObligationCause,
        param_env: ParamEnv<'db>,
        variables1: &OriginalQueryValues<'db>,
        variables2: impl Fn(BoundVar) -> GenericArg<'db>,
    ) -> InferResult<'db, ()> {
        let mut obligations = PredicateObligations::new();
        for (index, value1) in variables1.var_values.iter().enumerate() {
            let value2 = variables2(BoundVar::new(index));

            match (value1.kind(), value2.kind()) {
                (GenericArgKind::Type(v1), GenericArgKind::Type(v2)) => {
                    obligations.extend(self.at(cause, param_env).eq(v1, v2)?.into_obligations());
                }
                (GenericArgKind::Lifetime(re1), GenericArgKind::Lifetime(re2))
                    if re1.is_erased() && re2.is_erased() =>
                {
                    // no action needed
                }
                (GenericArgKind::Lifetime(v1), GenericArgKind::Lifetime(v2)) => {
                    self.inner.borrow_mut().unwrap_region_constraints().make_eqregion(v1, v2);
                }
                (GenericArgKind::Const(v1), GenericArgKind::Const(v2)) => {
                    let ok = self.at(cause, param_env).eq(v1, v2)?;
                    obligations.extend(ok.into_obligations());
                }
                _ => {
                    panic!("kind mismatch, cannot unify {:?} and {:?}", value1, value2,);
                }
            }
        }
        Ok(InferOk { value: (), obligations })
    }
}
