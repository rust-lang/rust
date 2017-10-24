// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::mir::Mir;
use rustc::infer::InferCtxt;

use super::LivenessResults;
use super::ToRegionIndex;
use super::region_infer::RegionInferenceContext;

pub(super) fn generate_constraints<'a, 'gcx, 'tcx>(
    infcx: &InferCtxt<'a, 'gcx, 'tcx>,
    regioncx: &mut RegionInferenceContext,
    mir: &Mir<'tcx>,
    liveness: &LivenessResults,
) {
    ConstraintGeneration {
        infcx,
        regioncx,
        mir,
        liveness,
    }.add_constraints();
}

struct ConstraintGeneration<'constrain, 'gcx: 'tcx, 'tcx: 'constrain> {
    infcx: &'constrain InferCtxt<'constrain, 'gcx, 'tcx>,
    regioncx: &'constrain mut RegionInferenceContext,
    mir: &'constrain Mir<'tcx>,
    liveness: &'constrain LivenessResults,
}

impl<'constrain, 'gcx, 'tcx> ConstraintGeneration<'constrain, 'gcx, 'tcx> {
    fn add_constraints(&mut self) {
        // To start, add the liveness constraints.
        self.add_liveness_constraints();
    }

    /// Liveness constraints:
    ///
    /// > If a variable V is live at point P, then all regions R in the type of V
    /// > must include the point P.
    fn add_liveness_constraints(&mut self) {
        let tcx = self.infcx.tcx;

        debug!("add_liveness_constraints()");
        for bb in self.mir.basic_blocks().indices() {
            debug!("add_liveness_constraints: bb={:?}", bb);

            self.liveness
                .regular
                .simulate_block(self.mir, bb, |location, live_locals| {
                    debug!(
                        "add_liveness_constraints: location={:?} live_locals={:?}",
                        location,
                        live_locals
                    );

                    for live_local in live_locals.iter() {
                        let live_local_ty = self.mir.local_decls[live_local].ty;
                        tcx.for_each_free_region(&live_local_ty, |live_region| {
                            let vid = live_region.to_region_index();
                            self.regioncx.add_live_point(vid, location);
                        })
                    }
                });
        }
    }
}
