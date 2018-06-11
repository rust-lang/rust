// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use infer::canonical::query_result;
use infer::canonical::{Canonicalized, CanonicalizedQueryResult, QueryRegionConstraint};
use infer::{InferCtxt, InferOk, InferResult};
use traits::{ObligationCause, TraitEngine};
use ty::error::TypeError;
use ty::fold::TypeFoldable;
use ty::{Lift, ParamEnv, TyCtxt};
use std::fmt;
use std::rc::Rc;
use syntax::codemap::DUMMY_SP;

pub mod custom;
pub mod eq;
pub mod normalize;
pub mod outlives;
pub mod predicates;
pub mod subtype;

pub trait TypeOp<'gcx, 'tcx>: Sized + fmt::Debug {
    type Output;

    /// Micro-optimization: returns `Ok(x)` if we can trivially
    /// produce the output, else returns `Err(self)` back.
    fn trivial_noop(self, tcx: TyCtxt<'_, 'gcx, 'tcx>) -> Result<Self::Output, Self>;

    /// Given an infcx, performs **the kernel** of the operation: this does the
    /// key action and then, optionally, returns a set of obligations which must be proven.
    ///
    /// This method is not meant to be invoked directly: instead, one
    /// should use `fully_perform`, which will take those resulting
    /// obligations and prove them, and then process the combined
    /// results into region obligations which are returned.
    fn perform(self, infcx: &InferCtxt<'_, 'gcx, 'tcx>) -> InferResult<'tcx, Self::Output>;

    /// Processes the operation and all resulting obligations,
    /// returning the final result along with any region constraints
    /// (they will be given over to the NLL region solver).
    fn fully_perform(
        self,
        infcx: &InferCtxt<'_, 'gcx, 'tcx>,
    ) -> Result<(Self::Output, Option<Rc<Vec<QueryRegionConstraint<'tcx>>>>), TypeError<'tcx>> {
        match self.trivial_noop(infcx.tcx) {
            Ok(r) => Ok((r, None)),
            Err(op) => op.fully_perform_nontrivial(infcx),
        }
    }

    /// Helper for `fully_perform` that handles the nontrivial cases.
    #[inline(never)] // just to help with profiling
    fn fully_perform_nontrivial(
        self,
        infcx: &InferCtxt<'_, 'gcx, 'tcx>,
    ) -> Result<(Self::Output, Option<Rc<Vec<QueryRegionConstraint<'tcx>>>>), TypeError<'tcx>> {
        if cfg!(debug_assertions) {
            info!(
                "fully_perform_op_and_get_region_constraint_data({:?})",
                self
            );
        }

        let mut fulfill_cx = TraitEngine::new(infcx.tcx);
        let dummy_body_id = ObligationCause::dummy().body_id;
        let InferOk { value, obligations } = infcx.commit_if_ok(|_| self.perform(infcx))?;
        debug_assert!(obligations.iter().all(|o| o.cause.body_id == dummy_body_id));
        fulfill_cx.register_predicate_obligations(infcx, obligations);
        if let Err(e) = fulfill_cx.select_all_or_error(infcx) {
            infcx.tcx.sess.diagnostic().delay_span_bug(
                DUMMY_SP,
                &format!("errors selecting obligation during MIR typeck: {:?}", e),
            );
        }

        let region_obligations = infcx.take_registered_region_obligations();

        let region_constraint_data = infcx.take_and_reset_region_constraints();

        let outlives = query_result::make_query_outlives(
            infcx.tcx,
            region_obligations,
            &region_constraint_data,
        );

        if outlives.is_empty() {
            Ok((value, None))
        } else {
            Ok((value, Some(Rc::new(outlives))))
        }
    }
}

type Lifted<'gcx, T> = <T as Lift<'gcx>>::Lifted;

pub trait QueryTypeOp<'gcx: 'tcx, 'tcx>: TypeFoldable<'tcx> + Lift<'gcx> {
    type QueryResult: TypeFoldable<'tcx> + Lift<'gcx>;

    /// Micro-optimization: returns `Ok(x)` if we can trivially
    /// produce the output, else returns `Err(self)` back.
    fn trivial_noop(
        self,
        tcx: TyCtxt<'_, 'gcx, 'tcx>,
    ) -> Result<Lifted<'gcx, Self::QueryResult>, Self>;

    fn param_env(&self) -> ParamEnv<'tcx>;

    fn perform_query(
        tcx: TyCtxt<'_, 'gcx, 'tcx>,
        canonicalized: Canonicalized<'gcx, Self>,
    ) -> CanonicalizedQueryResult<'gcx, Self::QueryResult>;
}

impl<'gcx: 'tcx, 'tcx, Q> TypeOp<'gcx, 'tcx> for Q
where
    Q: QueryTypeOp<'gcx, 'tcx>,
    Lifted<'gcx, Q::QueryResult>: TypeFoldable<'tcx>,
{
    type Output = Lifted<'gcx, Q::QueryResult>;

    fn trivial_noop(self, tcx: TyCtxt<'_, 'gcx, 'tcx>) -> Result<Self::Output, Self> {
        QueryTypeOp::trivial_noop(self, tcx)
    }

    fn perform(self, infcx: &InferCtxt<'_, 'gcx, 'tcx>) -> InferResult<'tcx, Self::Output> {
        let param_env = self.param_env();

        let (canonical_self, canonical_var_values) = infcx.canonicalize_query(&self);
        let canonical_result = Q::perform_query(infcx.tcx, canonical_self);

        // FIXME: This is not the most efficient setup. The
        // `instantiate_query_result_and_region_obligations` basically
        // takes the `QueryRegionConstraint` values that we ultimately
        // want to use and converts them into obligations. We return
        // those to our caller, which will convert them into AST
        // region constraints; we then convert *those* back into
        // `QueryRegionConstraint` and ultimately into NLL
        // constraints. We should cut out the middleman but that will
        // take a bit of refactoring.
        infcx.instantiate_query_result_and_region_obligations(
            &ObligationCause::dummy(),
            param_env,
            &canonical_var_values,
            &canonical_result,
        )
    }
}
