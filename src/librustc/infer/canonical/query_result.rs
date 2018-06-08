// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module contains the code to instantiate a "query result", and
//! in particular to extract out the resulting region obligations and
//! encode them therein.
//!
//! For an overview of what canonicaliation is and how it fits into
//! rustc, check out the [chapter in the rustc guide][c].
//!
//! [c]: https://rust-lang-nursery.github.io/rustc-guide/traits/canonicalization.html

use infer::canonical::substitute::substitute_value;
use infer::canonical::{
    Canonical, CanonicalVarValues, Canonicalize, Certainty, QueryRegionConstraint, QueryResult,
};
use infer::region_constraints::{Constraint, RegionConstraintData};
use infer::{InferCtxt, InferOk, InferResult};
use rustc_data_structures::indexed_vec::Idx;
use std::fmt::Debug;
use traits::query::NoSolution;
use traits::{FulfillmentContext, TraitEngine};
use traits::{Obligation, ObligationCause, PredicateObligation};
use ty::fold::TypeFoldable;
use ty::subst::{Kind, UnpackedKind};
use ty::{self, CanonicalVar};

use rustc_data_structures::indexed_vec::IndexVec;

type CanonicalizedQueryResult<'gcx, 'tcx, T> =
    <QueryResult<'tcx, T> as Canonicalize<'gcx, 'tcx>>::Canonicalized;

impl<'cx, 'gcx, 'tcx> InferCtxt<'cx, 'gcx, 'tcx> {
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
    pub fn make_canonicalized_query_result<T>(
        &self,
        inference_vars: CanonicalVarValues<'tcx>,
        answer: T,
        fulfill_cx: &mut FulfillmentContext<'tcx>,
    ) -> Result<CanonicalizedQueryResult<'gcx, 'tcx, T>, NoSolution>
    where
        T: Debug,
        QueryResult<'tcx, T>: Canonicalize<'gcx, 'tcx>,
    {
        let tcx = self.tcx;

        debug!(
            "make_query_response(\
             inference_vars={:?}, \
             answer={:?})",
            inference_vars, answer,
        );

        // Select everything, returning errors.
        let true_errors = match fulfill_cx.select_where_possible(self) {
            Ok(()) => vec![],
            Err(errors) => errors,
        };
        debug!("true_errors = {:#?}", true_errors);

        if !true_errors.is_empty() {
            // FIXME -- we don't indicate *why* we failed to solve
            debug!("make_query_response: true_errors={:#?}", true_errors);
            return Err(NoSolution);
        }

        // Anything left unselected *now* must be an ambiguity.
        let ambig_errors = match fulfill_cx.select_all_or_error(self) {
            Ok(()) => vec![],
            Err(errors) => errors,
        };
        debug!("ambig_errors = {:#?}", ambig_errors);

        let region_obligations = self.take_registered_region_obligations();

        let region_constraints = self.with_region_constraints(|region_constraints| {
            let RegionConstraintData {
                constraints,
                verifys,
                givens,
            } = region_constraints;

            assert!(verifys.is_empty());
            assert!(givens.is_empty());

            let mut outlives: Vec<_> = constraints
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
            .map(ty::Binder::dummy) // no bound regions in the code above
            .collect();

            outlives.extend(
                region_obligations
                    .into_iter()
                    .map(|(_, r_o)| ty::OutlivesPredicate(r_o.sup_type.into(), r_o.sub_region))
                    .map(ty::Binder::dummy), // no bound regions in the code above
            );

            outlives
        });

        let certainty = if ambig_errors.is_empty() {
            Certainty::Proven
        } else {
            Certainty::Ambiguous
        };

        let (canonical_result, _) = self.canonicalize_response(&QueryResult {
            var_values: inference_vars,
            region_constraints,
            certainty,
            value: answer,
        });

        debug!(
            "make_query_response: canonical_result = {:#?}",
            canonical_result
        );

        Ok(canonical_result)
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
    /// [c]: https://rust-lang-nursery.github.io/rustc-guide/traits/canonicalization.html#processing-the-canonicalized-query-result
    pub fn instantiate_query_result<R>(
        &self,
        cause: &ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        original_values: &CanonicalVarValues<'tcx>,
        query_result: &Canonical<'tcx, QueryResult<'tcx, R>>,
    ) -> InferResult<'tcx, R>
    where
        R: Debug + TypeFoldable<'tcx>,
    {
        debug!(
            "instantiate_query_result(original_values={:#?}, query_result={:#?})",
            original_values, query_result,
        );

        // Every canonical query result includes values for each of
        // the inputs to the query. Therefore, we begin by unifying
        // these values with the original inputs that were
        // canonicalized.
        let result_values = &query_result.value.var_values;
        assert_eq!(original_values.len(), result_values.len());

        // Quickly try to find initial values for the canonical
        // variables in the result in terms of the query. We do this
        // by iterating down the values that the query gave to each of
        // the canonical inputs. If we find that one of those values
        // is directly equal to one of the canonical variables in the
        // result, then we can type the corresponding value from the
        // input. See the example above.
        let mut opt_values: IndexVec<CanonicalVar, Option<Kind<'tcx>>> =
            IndexVec::from_elem_n(None, query_result.variables.len());

        // In terms of our example above, we are iterating over pairs like:
        // [(?A, Vec<?0>), ('static, '?1), (?B, ?0)]
        for (original_value, result_value) in original_values.iter().zip(result_values) {
            match result_value.unpack() {
                UnpackedKind::Type(result_value) => {
                    // e.g., here `result_value` might be `?0` in the example above...
                    if let ty::TyInfer(ty::InferTy::CanonicalTy(index)) = result_value.sty {
                        // in which case we would set `canonical_vars[0]` to `Some(?U)`.
                        opt_values[index] = Some(original_value);
                    }
                }
                UnpackedKind::Lifetime(result_value) => {
                    // e.g., here `result_value` might be `'?1` in the example above...
                    if let &ty::RegionKind::ReCanonical(index) = result_value {
                        // in which case we would set `canonical_vars[0]` to `Some('static)`.
                        opt_values[index] = Some(original_value);
                    }
                }
            }
        }

        // Create a result substitution: if we found a value for a
        // given variable in the loop above, use that. Otherwise, use
        // a fresh inference variable.
        let result_subst = &CanonicalVarValues {
            var_values: query_result
                .variables
                .iter()
                .enumerate()
                .map(|(index, info)| match opt_values[CanonicalVar::new(index)] {
                    Some(k) => k,
                    None => self.fresh_inference_var_for_canonical_var(cause.span, *info),
                })
                .collect(),
        };

        // Unify the original values for the canonical variables in
        // the input with the value found in the query
        // post-substitution. Often, but not always, this is a no-op,
        // because we already found the mapping in the first step.
        let substituted_values = |index: CanonicalVar| -> Kind<'tcx> {
            query_result.substitute_projected(self.tcx, result_subst, |v| &v.var_values[index])
        };
        let mut obligations = self
            .unify_canonical_vars(cause, param_env, original_values, substituted_values)?
            .into_obligations();

        obligations.extend(self.query_region_constraints_into_obligations(
            cause,
            param_env,
            &query_result.value.region_constraints,
            result_subst,
        ));

        let user_result: R =
            query_result.substitute_projected(self.tcx, result_subst, |q_r| &q_r.value);

        Ok(InferOk {
            value: user_result,
            obligations,
        })
    }

    /// Converts the region constraints resulting from a query into an
    /// iterator of obligations.
    fn query_region_constraints_into_obligations<'a>(
        &'a self,
        cause: &'a ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        unsubstituted_region_constraints: &'a [QueryRegionConstraint<'tcx>],
        result_subst: &'a CanonicalVarValues<'tcx>,
    ) -> impl Iterator<Item = PredicateObligation<'tcx>> + 'a {
        Box::new(
            unsubstituted_region_constraints
                .iter()
                .map(move |constraint| {
                    let ty::OutlivesPredicate(k1, r2) = constraint.skip_binder(); // restored below
                    let k1 = substitute_value(self.tcx, result_subst, k1);
                    let r2 = substitute_value(self.tcx, result_subst, r2);
                    match k1.unpack() {
                        UnpackedKind::Lifetime(r1) => Obligation::new(
                            cause.clone(),
                            param_env,
                            ty::Predicate::RegionOutlives(ty::Binder::dummy(
                                ty::OutlivesPredicate(r1, r2),
                            )),
                        ),

                        UnpackedKind::Type(t1) => Obligation::new(
                            cause.clone(),
                            param_env,
                            ty::Predicate::TypeOutlives(ty::Binder::dummy(ty::OutlivesPredicate(
                                t1, r2,
                            ))),
                        ),
                    }
                }),
        ) as Box<dyn Iterator<Item = _>>
    }

    /// Given two sets of values for the same set of canonical variables, unify them.
    /// The second set is produced lazilly by supplying indices from the first set.
    fn unify_canonical_vars(
        &self,
        cause: &ObligationCause<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        variables1: &CanonicalVarValues<'tcx>,
        variables2: impl Fn(CanonicalVar) -> Kind<'tcx>,
    ) -> InferResult<'tcx, ()> {
        self.commit_if_ok(|_| {
            let mut obligations = vec![];
            for (index, value1) in variables1.var_values.iter_enumerated() {
                let value2 = variables2(index);

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
