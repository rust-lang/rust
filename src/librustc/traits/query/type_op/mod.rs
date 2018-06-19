// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use infer::canonical::{
    Canonical, Canonicalized, CanonicalizedQueryResult, QueryRegionConstraint, QueryResult,
};
use infer::{InferCtxt, InferOk};
use std::fmt;
use std::rc::Rc;
use traits::query::Fallible;
use traits::ObligationCause;
use ty::fold::TypeFoldable;
use ty::{Lift, ParamEnv, TyCtxt};

pub mod custom;
pub mod eq;
pub mod normalize;
pub mod outlives;
pub mod prove_predicate;
pub mod subtype;

pub trait TypeOp<'gcx, 'tcx>: Sized + fmt::Debug {
    type Output;

    /// Processes the operation and all resulting obligations,
    /// returning the final result along with any region constraints
    /// (they will be given over to the NLL region solver).
    fn fully_perform(
        self,
        infcx: &InferCtxt<'_, 'gcx, 'tcx>,
    ) -> Fallible<(Self::Output, Option<Rc<Vec<QueryRegionConstraint<'tcx>>>>)>;
}

pub trait QueryTypeOp<'gcx: 'tcx, 'tcx>: fmt::Debug + Sized {
    type QueryKey: TypeFoldable<'tcx> + Lift<'gcx>;
    type QueryResult: TypeFoldable<'tcx> + Lift<'gcx>;

    /// Either converts `self` directly into a `QueryResult` (for
    /// simple cases) or into a `QueryKey` (for more complex cases
    /// where we actually have work to do).
    fn prequery(self, tcx: TyCtxt<'_, 'gcx, 'tcx>) -> Result<Self::QueryResult, Self::QueryKey>;

    fn param_env(key: &Self::QueryKey) -> ParamEnv<'tcx>;

    fn perform_query(
        tcx: TyCtxt<'_, 'gcx, 'tcx>,
        canonicalized: Canonicalized<'gcx, Self::QueryKey>,
    ) -> Fallible<CanonicalizedQueryResult<'gcx, Self::QueryResult>>;

    /// "Upcasts" a lifted query result (which is in the gcx lifetime)
    /// into the tcx lifetime. This is always just an identity cast,
    /// but the generic code does't realize it, so we have to push the
    /// operation into the impls that know more specifically what
    /// `QueryResult` is. This operation would (maybe) be nicer with
    /// something like HKTs or GATs, since then we could make
    /// `QueryResult` parametric and `'gcx` and `'tcx` etc.
    fn upcast_result(
        lifted_query_result: &'a CanonicalizedQueryResult<'gcx, Self::QueryResult>,
    ) -> &'a Canonical<'tcx, QueryResult<'tcx, Self::QueryResult>>;

    fn fully_perform_into(
        self,
        infcx: &InferCtxt<'_, 'gcx, 'tcx>,
        output_query_region_constraints: &mut Vec<QueryRegionConstraint<'tcx>>,
    ) -> Fallible<Self::QueryResult> {
        match QueryTypeOp::prequery(self, infcx.tcx) {
            Ok(result) => Ok(result),
            Err(query_key) => {
                // FIXME(#33684) -- We need to use
                // `canonicalize_hr_query_hack` here because of things
                // like the subtype query, which go awry around
                // `'static` otherwise.
                let (canonical_self, canonical_var_values) =
                    infcx.canonicalize_hr_query_hack(&query_key);
                let canonical_result = Self::perform_query(infcx.tcx, canonical_self)?;
                let canonical_result = Self::upcast_result(&canonical_result);

                let param_env = Self::param_env(&query_key);

                let InferOk { value, obligations } = infcx
                    .instantiate_nll_query_result_and_region_obligations(
                        &ObligationCause::dummy(),
                        param_env,
                        &canonical_var_values,
                        canonical_result,
                        output_query_region_constraints,
                    )?;

                // Typically, instantiating NLL query results does not
                // create obligations. However, in some cases there
                // are unresolved type variables, and unify them *can*
                // create obligations. In that case, we have to go
                // fulfill them. We do this via a (recursive) query.
                for obligation in obligations {
                    let () = prove_predicate::ProvePredicate::new(
                        obligation.param_env,
                        obligation.predicate,
                    ).fully_perform_into(infcx, output_query_region_constraints)?;
                }

                Ok(value)
            }
        }
    }
}

impl<'gcx: 'tcx, 'tcx, Q> TypeOp<'gcx, 'tcx> for Q
where
    Q: QueryTypeOp<'gcx, 'tcx>,
{
    type Output = Q::QueryResult;

    fn fully_perform(
        self,
        infcx: &InferCtxt<'_, 'gcx, 'tcx>,
    ) -> Fallible<(Self::Output, Option<Rc<Vec<QueryRegionConstraint<'tcx>>>>)> {
        let mut qrc = vec![];
        let r = Q::fully_perform_into(self, infcx, &mut qrc)?;

        // Promote the final query-region-constraints into a
        // (optional) ref-counted vector:
        let opt_qrc = if qrc.is_empty() {
            None
        } else {
            Some(Rc::new(qrc))
        };

        Ok((r, opt_qrc))
    }
}
