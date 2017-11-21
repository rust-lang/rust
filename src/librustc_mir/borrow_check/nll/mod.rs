// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::hir::def_id::DefId;
use rustc::mir::Mir;
use rustc::infer::InferCtxt;
use rustc::ty::{self, RegionKind, RegionVid};
use rustc::util::nodemap::FxHashMap;
use std::collections::BTreeSet;
use transform::MirSource;
use transform::type_check;
use util::liveness::{self, LivenessMode, LivenessResult, LocalSet};
use borrow_check::FlowInProgress;
use dataflow::MaybeInitializedLvals;
use dataflow::move_paths::MoveData;

use util as mir_util;
use self::mir_util::PassWhere;

mod constraint_generation;
mod subtype_constraint_generation;
mod universal_regions;
use self::universal_regions::UniversalRegions;

pub(crate) mod region_infer;
use self::region_infer::RegionInferenceContext;

mod renumber;

/// Rewrites the regions in the MIR to use NLL variables, also
/// scraping out the set of free regions (e.g., region parameters)
/// declared on the function. That set will need to be given to
/// `compute_regions`.
pub(in borrow_check) fn replace_regions_in_mir<'cx, 'gcx, 'tcx>(
    infcx: &InferCtxt<'cx, 'gcx, 'tcx>,
    def_id: DefId,
    mir: &mut Mir<'tcx>,
) -> UniversalRegions<'tcx> {
    // Compute named region information.
    let universal_regions = universal_regions::universal_regions(infcx, def_id);

    // Replace all regions with fresh inference variables.
    renumber::renumber_mir(infcx, &universal_regions, mir);

    universal_regions
}

/// Computes the (non-lexical) regions from the input MIR.
///
/// This may result in errors being reported.
pub(in borrow_check) fn compute_regions<'cx, 'gcx, 'tcx>(
    infcx: &InferCtxt<'cx, 'gcx, 'tcx>,
    def_id: DefId,
    universal_regions: UniversalRegions<'tcx>,
    mir: &Mir<'tcx>,
    param_env: ty::ParamEnv<'gcx>,
    flow_inits: &mut FlowInProgress<MaybeInitializedLvals<'cx, 'gcx, 'tcx>>,
    move_data: &MoveData<'tcx>,
) -> RegionInferenceContext<'tcx> {
    // Run the MIR type-checker.
    let mir_node_id = infcx.tcx.hir.as_local_node_id(def_id).unwrap();
    let constraint_sets = &type_check::type_check(infcx, mir_node_id, param_env, mir);

    // Create the region inference context, taking ownership of the region inference
    // data that was contained in `infcx`.
    let var_origins = infcx.take_region_var_origins();
    let mut regioncx = RegionInferenceContext::new(var_origins, &universal_regions, mir);
    subtype_constraint_generation::generate(
        &mut regioncx,
        &universal_regions,
        mir,
        constraint_sets,
    );

    // Compute what is live where.
    let liveness = &LivenessResults {
        regular: liveness::liveness_of_locals(
            &mir,
            LivenessMode {
                include_regular_use: true,
                include_drops: false,
            },
        ),

        drop: liveness::liveness_of_locals(
            &mir,
            LivenessMode {
                include_regular_use: false,
                include_drops: true,
            },
        ),
    };

    // Generate non-subtyping constraints.
    constraint_generation::generate_constraints(
        infcx,
        &mut regioncx,
        &mir,
        param_env,
        liveness,
        flow_inits,
        move_data,
    );

    // Solve the region constraints.
    regioncx.solve(infcx, &mir);

    // Dump MIR results into a file, if that is enabled. This let us
    // write unit-tests.
    dump_mir_results(infcx, liveness, MirSource::item(def_id), &mir, &regioncx);

    regioncx
}

struct LivenessResults {
    regular: LivenessResult,
    drop: LivenessResult,
}

fn dump_mir_results<'a, 'gcx, 'tcx>(
    infcx: &InferCtxt<'a, 'gcx, 'tcx>,
    liveness: &LivenessResults,
    source: MirSource,
    mir: &Mir<'tcx>,
    regioncx: &RegionInferenceContext,
) {
    if !mir_util::dump_enabled(infcx.tcx, "nll", source) {
        return;
    }

    let regular_liveness_per_location: FxHashMap<_, _> = mir.basic_blocks()
        .indices()
        .flat_map(|bb| {
            let mut results = vec![];
            liveness
                .regular
                .simulate_block(&mir, bb, |location, local_set| {
                    results.push((location, local_set.clone()));
                });
            results
        })
        .collect();

    let drop_liveness_per_location: FxHashMap<_, _> = mir.basic_blocks()
        .indices()
        .flat_map(|bb| {
            let mut results = vec![];
            liveness
                .drop
                .simulate_block(&mir, bb, |location, local_set| {
                    results.push((location, local_set.clone()));
                });
            results
        })
        .collect();

    mir_util::dump_mir(infcx.tcx, None, "nll", &0, source, mir, |pass_where, out| {
        match pass_where {
            // Before the CFG, dump out the values for each region variable.
            PassWhere::BeforeCFG => for region in regioncx.regions() {
                writeln!(out, "| {:?}: {}", region, regioncx.region_value_str(region))?;
            },

            // Before each basic block, dump out the values
            // that are live on entry to the basic block.
            PassWhere::BeforeBlock(bb) => {
                let s = live_variable_set(&liveness.regular.ins[bb], &liveness.drop.ins[bb]);
                writeln!(out, "    | Live variables on entry to {:?}: {}", bb, s)?;
            }

            PassWhere::BeforeLocation(location) => {
                let s = live_variable_set(
                    &regular_liveness_per_location[&location],
                    &drop_liveness_per_location[&location],
                );
                writeln!(out, "            | Live variables at {:?}: {}", location, s)?;
            }

            PassWhere::AfterLocation(_) | PassWhere::AfterCFG => {}
        }
        Ok(())
    });
}

/// Right now, we piggy back on the `ReVar` to store our NLL inference
/// regions. These are indexed with `RegionVid`. This method will
/// assert that the region is a `ReVar` and extract its interal index.
/// This is reasonable because in our MIR we replace all free regions
/// with inference variables.
pub trait ToRegionVid {
    fn to_region_vid(&self) -> RegionVid;
}

impl ToRegionVid for RegionKind {
    fn to_region_vid(&self) -> RegionVid {
        if let &ty::ReVar(vid) = self {
            vid
        } else {
            bug!("region is not an ReVar: {:?}", self)
        }
    }
}

fn live_variable_set(regular: &LocalSet, drops: &LocalSet) -> String {
    // sort and deduplicate:
    let all_locals: BTreeSet<_> = regular.iter().chain(drops.iter()).collect();

    // construct a string with each local, including `(drop)` if it is
    // only dropped, versus a regular use.
    let mut string = String::new();
    for local in all_locals {
        string.push_str(&format!("{:?}", local));

        if !regular.contains(&local) {
            assert!(drops.contains(&local));
            string.push_str(" (drop)");
        }

        string.push_str(", ");
    }

    let len = if string.is_empty() {
        0
    } else {
        string.len() - 2
    };

    format!("[{}]", &string[..len])
}
