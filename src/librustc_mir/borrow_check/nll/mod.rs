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
use rustc::mir::{ClosureRegionRequirements, ClosureOutlivesSubject, Mir};
use rustc::infer::InferCtxt;
use rustc::ty::{self, RegionKind, RegionVid};
use rustc::util::nodemap::FxHashMap;
use std::collections::BTreeSet;
use std::fmt::Debug;
use std::io;
use transform::MirSource;
use util::liveness::{LivenessResults, LocalSet};
use dataflow::FlowAtLocation;
use dataflow::MaybeInitializedLvals;
use dataflow::move_paths::MoveData;

use util as mir_util;
use util::pretty::{self, ALIGN};
use self::mir_util::PassWhere;

mod constraint_generation;
pub mod explain_borrow;
pub(crate) mod region_infer;
mod renumber;
mod subtype_constraint_generation;
pub(crate) mod type_check;
mod universal_regions;

use self::region_infer::RegionInferenceContext;
use self::universal_regions::UniversalRegions;


/// Rewrites the regions in the MIR to use NLL variables, also
/// scraping out the set of universal regions (e.g., region parameters)
/// declared on the function. That set will need to be given to
/// `compute_regions`.
pub(in borrow_check) fn replace_regions_in_mir<'cx, 'gcx, 'tcx>(
    infcx: &InferCtxt<'cx, 'gcx, 'tcx>,
    def_id: DefId,
    param_env: ty::ParamEnv<'tcx>,
    mir: &mut Mir<'tcx>,
) -> UniversalRegions<'tcx> {
    debug!("replace_regions_in_mir(def_id={:?})", def_id);

    // Compute named region information. This also renumbers the inputs/outputs.
    let universal_regions = UniversalRegions::new(infcx, def_id, param_env);

    // Replace all remaining regions with fresh inference variables.
    renumber::renumber_mir(infcx, mir);

    let source = MirSource::item(def_id);
    mir_util::dump_mir(infcx.tcx, None, "renumber", &0, source, mir, |_, _| Ok(()));

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
    flow_inits: &mut FlowAtLocation<MaybeInitializedLvals<'cx, 'gcx, 'tcx>>,
    move_data: &MoveData<'tcx>,
) -> (
    RegionInferenceContext<'tcx>,
    Option<ClosureRegionRequirements<'gcx>>,
) {
    // Run the MIR type-checker.
    let liveness = &LivenessResults::compute(mir);
    let constraint_sets = &type_check::type_check(
        infcx,
        param_env,
        mir,
        def_id,
        &universal_regions,
        &liveness,
        flow_inits,
        move_data,
    );

    // Create the region inference context, taking ownership of the region inference
    // data that was contained in `infcx`.
    let var_origins = infcx.take_region_var_origins();
    let mut regioncx = RegionInferenceContext::new(var_origins, universal_regions, mir);
    subtype_constraint_generation::generate(&mut regioncx, mir, constraint_sets);


    // Generate non-subtyping constraints.
    constraint_generation::generate_constraints(infcx, &mut regioncx, &mir);

    // Solve the region constraints.
    let closure_region_requirements = regioncx.solve(infcx, &mir, def_id);

    // Dump MIR results into a file, if that is enabled. This let us
    // write unit-tests, as well as helping with debugging.
    dump_mir_results(
        infcx,
        liveness,
        MirSource::item(def_id),
        &mir,
        &regioncx,
        &closure_region_requirements,
    );

    // We also have a `#[rustc_nll]` annotation that causes us to dump
    // information
    dump_annotation(infcx, &mir, def_id, &regioncx, &closure_region_requirements);

    (regioncx, closure_region_requirements)
}

fn dump_mir_results<'a, 'gcx, 'tcx>(
    infcx: &InferCtxt<'a, 'gcx, 'tcx>,
    liveness: &LivenessResults,
    source: MirSource,
    mir: &Mir<'tcx>,
    regioncx: &RegionInferenceContext,
    closure_region_requirements: &Option<ClosureRegionRequirements>,
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
            PassWhere::BeforeCFG => {
                regioncx.dump_mir(out)?;

                if let Some(closure_region_requirements) = closure_region_requirements {
                    writeln!(out, "|")?;
                    writeln!(out, "| Free Region Constraints")?;
                    for_each_region_constraint(closure_region_requirements, &mut |msg| {
                        writeln!(out, "| {}", msg)
                    })?;
                }
            }

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
                writeln!(
                    out,
                    "{:ALIGN$} | Live variables on entry to {:?}: {}",
                    "",
                    location,
                    s,
                    ALIGN = ALIGN
                )?;
            }

            PassWhere::AfterLocation(_) | PassWhere::AfterCFG => {}
        }
        Ok(())
    });

    // Also dump the inference graph constraints as a graphviz file.
    let _: io::Result<()> = do catch {
        let mut file =
            pretty::create_dump_file(infcx.tcx, "regioncx.dot", None, "nll", &0, source)?;
        regioncx.dump_graphviz(&mut file)
    };
}

fn dump_annotation<'a, 'gcx, 'tcx>(
    infcx: &InferCtxt<'a, 'gcx, 'tcx>,
    mir: &Mir<'tcx>,
    mir_def_id: DefId,
    regioncx: &RegionInferenceContext,
    closure_region_requirements: &Option<ClosureRegionRequirements>,
) {
    let tcx = infcx.tcx;
    let base_def_id = tcx.closure_base_def_id(mir_def_id);
    if !tcx.has_attr(base_def_id, "rustc_regions") {
        return;
    }

    // When the enclosing function is tagged with `#[rustc_regions]`,
    // we dump out various bits of state as warnings. This is useful
    // for verifying that the compiler is behaving as expected.  These
    // warnings focus on the closure region requirements -- for
    // viewing the intraprocedural state, the -Zdump-mir output is
    // better.

    if let Some(closure_region_requirements) = closure_region_requirements {
        let mut err = tcx.sess
            .diagnostic()
            .span_note_diag(mir.span, "External requirements");

        regioncx.annotate(&mut err);

        err.note(&format!(
            "number of external vids: {}",
            closure_region_requirements.num_external_vids
        ));

        // Dump the region constraints we are imposing *between* those
        // newly created variables.
        for_each_region_constraint(closure_region_requirements, &mut |msg| {
            err.note(msg);
            Ok(())
        }).unwrap();

        err.emit();
    } else {
        let mut err = tcx.sess
            .diagnostic()
            .span_note_diag(mir.span, "No external requirements");
        regioncx.annotate(&mut err);
        err.emit();
    }
}

fn for_each_region_constraint(
    closure_region_requirements: &ClosureRegionRequirements,
    with_msg: &mut FnMut(&str) -> io::Result<()>,
) -> io::Result<()> {
    for req in &closure_region_requirements.outlives_requirements {
        let subject: &Debug = match &req.subject {
            ClosureOutlivesSubject::Region(subject) => subject,
            ClosureOutlivesSubject::Ty(ty) => ty,
        };
        with_msg(&format!(
            "where {:?}: {:?}",
            subject,
            req.outlived_free_region,
        ))?;
    }
    Ok(())
}

/// Right now, we piggy back on the `ReVar` to store our NLL inference
/// regions. These are indexed with `RegionVid`. This method will
/// assert that the region is a `ReVar` and extract its interal index.
/// This is reasonable because in our MIR we replace all universal regions
/// with inference variables.
pub trait ToRegionVid {
    fn to_region_vid(self) -> RegionVid;
}

impl<'tcx> ToRegionVid for &'tcx RegionKind {
    fn to_region_vid(self) -> RegionVid {
        if let ty::ReVar(vid) = self {
            *vid
        } else {
            bug!("region is not an ReVar: {:?}", self)
        }
    }
}

impl ToRegionVid for RegionVid {
    fn to_region_vid(self) -> RegionVid {
        self
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
