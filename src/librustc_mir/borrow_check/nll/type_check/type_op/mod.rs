// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::infer::canonical::query_result;
use rustc::infer::canonical::QueryRegionConstraint;
use rustc::infer::{InferCtxt, InferOk, InferResult};
use rustc::traits::{ObligationCause, TraitEngine};
use rustc::ty::error::TypeError;
use rustc::ty::TyCtxt;
use std::fmt;
use std::rc::Rc;
use syntax::codemap::DUMMY_SP;

crate mod custom;
crate mod eq;
crate mod normalize;
crate mod predicates;
crate mod outlives;
crate mod subtype;

crate trait TypeOp<'gcx, 'tcx>: Sized + fmt::Debug {
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
