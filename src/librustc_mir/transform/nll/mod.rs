// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::ty::{self, RegionKind, TyCtxt};
use rustc::mir::{Location, Mir};
use rustc::mir::transform::{MirPass, MirSource};
use rustc::infer::InferCtxt;
use rustc::util::nodemap::FxHashMap;
use rustc_data_structures::indexed_vec::Idx;
use std::collections::BTreeSet;
use std::fmt;
use util::liveness::{self, LivenessResult};

use util as mir_util;
use self::mir_util::PassWhere;

mod constraint_generation;

mod region_infer;
use self::region_infer::RegionInferenceContext;

mod renumber;

// MIR Pass for non-lexical lifetimes
pub struct NLL;

impl MirPass for NLL {
    fn run_pass<'a, 'tcx>(
        &self,
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        source: MirSource,
        input_mir: &mut Mir<'tcx>,
    ) {
        if !tcx.sess.opts.debugging_opts.nll {
            return;
        }

        tcx.infer_ctxt().enter(|ref infcx| {
            // Clone mir so we can mutate it without disturbing the rest of the compiler
            let mir = &mut input_mir.clone();

            // Replace all regions with fresh inference variables.
            let num_region_variables = renumber::renumber_mir(infcx, mir);

            // Compute what is live where.
            let liveness = &liveness::liveness_of_locals(mir);

            // Create the region inference context, generate the constraints,
            // and then solve them.
            let regioncx = &mut RegionInferenceContext::new(num_region_variables);
            constraint_generation::generate_constraints(infcx, regioncx, mir, liveness);
            regioncx.solve(infcx, mir);

            // Dump MIR results into a file, if that is enabled.
            dump_mir_results(infcx, liveness, source, regioncx, mir);
        })
    }
}

fn dump_mir_results<'a, 'gcx, 'tcx>(
    infcx: &InferCtxt<'a, 'gcx, 'tcx>,
    liveness: &LivenessResult,
    source: MirSource,
    regioncx: &RegionInferenceContext,
    mir: &Mir<'tcx>,
) {
    if !mir_util::dump_enabled(infcx.tcx, "nll", source) {
        return;
    }

    let liveness_per_location: FxHashMap<_, _> = mir.basic_blocks()
        .indices()
        .flat_map(|bb| {
            let mut results = vec![];
            liveness.simulate_block(&mir, bb, |location, local_set| {
                results.push((location, local_set.clone()));
            });
            results
        })
        .collect();

    mir_util::dump_mir(infcx.tcx, None, "nll", &0, source, mir, |pass_where, out| {
        match pass_where {
            // Before the CFG, dump out the values for each region variable.
            PassWhere::BeforeCFG => for region in regioncx.regions() {
                writeln!(out, "| {:?}: {:?}", region, regioncx.region_value(region))?;
            },

            // Before each basic block, dump out the values
            // that are live on entry to the basic block.
            PassWhere::BeforeBlock(bb) => {
                let local_set = &liveness.ins[bb];
                writeln!(out, "    | Variables live on entry to the block {:?}:", bb)?;
                for local in local_set.iter() {
                    writeln!(out, "    | - {:?}", local)?;
                }
            }

            PassWhere::InCFG(location) => {
                let local_set = &liveness_per_location[&location];
                writeln!(out, "        | Live variables here: {:?}", local_set)?;
            }

            PassWhere::AfterCFG => {}
        }
        Ok(())
    });
}

#[derive(Clone, Default, PartialEq, Eq)]
pub struct Region {
    points: BTreeSet<Location>,
}

impl fmt::Debug for Region {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(formatter, "{:?}", self.points)
    }
}

impl Region {
    pub fn add_point(&mut self, point: Location) -> bool {
        self.points.insert(point)
    }

    pub fn may_contain(&self, point: Location) -> bool {
        self.points.contains(&point)
    }
}

newtype_index!(RegionIndex {
    DEBUG_NAME = "R",
});

/// Right now, we piggy back on the `ReVar` to store our NLL inference
/// regions. These are indexed with `RegionIndex`. This method will
/// assert that the region is a `ReVar` and convert the internal index
/// into a `RegionIndex`. This is reasonable because in our MIR we
/// replace all free regions with inference variables.
trait ToRegionIndex {
    fn to_region_index(&self) -> RegionIndex;
}

impl ToRegionIndex for RegionKind {
    fn to_region_index(&self) -> RegionIndex {
        if let &ty::ReVar(vid) = self {
            RegionIndex::new(vid.index as usize)
        } else {
            bug!("region is not an ReVar: {:?}", self)
        }
    }
}
