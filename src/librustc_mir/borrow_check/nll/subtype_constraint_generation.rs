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
use rustc::infer::region_constraints::Constraint;
use rustc::infer::region_constraints::RegionConstraintData;
use rustc::ty;
use transform::type_check::MirTypeckRegionConstraints;
use transform::type_check::OutlivesSet;

use super::region_infer::RegionInferenceContext;

/// When the MIR type-checker executes, it validates all the types in
/// the MIR, and in the process generates a set of constraints that
/// must hold regarding the regions in the MIR, along with locations
/// *where* they must hold. This code takes those constriants and adds
/// them into the NLL `RegionInferenceContext`.
pub(super) fn generate<'tcx>(
    regioncx: &mut RegionInferenceContext<'tcx>,
    mir: &Mir<'tcx>,
    constraints: &MirTypeckRegionConstraints<'tcx>,
) {
    SubtypeConstraintGenerator {
        regioncx,
        mir,
    }.generate(constraints);
}

struct SubtypeConstraintGenerator<'cx, 'tcx: 'cx> {
    regioncx: &'cx mut RegionInferenceContext<'tcx>,
    mir: &'cx Mir<'tcx>,
}

impl<'cx, 'tcx> SubtypeConstraintGenerator<'cx, 'tcx> {
    fn generate(&mut self, constraints: &MirTypeckRegionConstraints<'tcx>) {
        let MirTypeckRegionConstraints {
            liveness_set,
            outlives_sets,
        } = constraints;

        debug!(
            "generate(liveness_set={} items, outlives_sets={} items)",
            liveness_set.len(),
            outlives_sets.len()
        );

        for (region, location) in liveness_set {
            debug!("generate: {:#?} is live at {:#?}", region, location);
            let region_vid = self.to_region_vid(region);
            self.regioncx.add_live_point(region_vid, *location);
        }

        for OutlivesSet { locations, data } in outlives_sets {
            debug!("generate: constraints at: {:#?}", locations);
            let RegionConstraintData {
                constraints,
                verifys,
                givens,
            } = data;

            for constraint in constraints.keys() {
                debug!("generate: constraint: {:?}", constraint);
                let (a_vid, b_vid) = match constraint {
                    Constraint::VarSubVar(a_vid, b_vid) => (*a_vid, *b_vid),
                    Constraint::RegSubVar(a_r, b_vid) => (self.to_region_vid(a_r), *b_vid),
                    Constraint::VarSubReg(a_vid, b_r) => (*a_vid, self.to_region_vid(b_r)),
                    Constraint::RegSubReg(a_r, b_r) => {
                        (self.to_region_vid(a_r), self.to_region_vid(b_r))
                    }
                };

                // We have the constraint that `a_vid <= b_vid`. Add
                // `b_vid: a_vid` to our region checker. Note that we
                // reverse direction, because `regioncx` talks about
                // "outlives" (`>=`) whereas the region constraints
                // talk about `<=`.
                let span = self.mir.source_info(locations.from_location).span;
                self.regioncx
                    .add_outlives(span, b_vid, a_vid, locations.at_location);
            }

            assert!(verifys.is_empty(), "verifys not yet implemented");
            assert!(
                givens.is_empty(),
                "MIR type-checker does not use givens (thank goodness)"
            );
        }
    }

    fn to_region_vid(&self, r: ty::Region<'tcx>) -> ty::RegionVid {
        // Every region that we see in the constraints came from the
        // MIR or from the parameter environment. If the former, it
        // will be a region variable.  If the latter, it will be in
        // the set of universal regions *somewhere*.
        if let ty::ReVar(vid) = r {
            *vid
        } else {
            self.regioncx.to_region_vid(r)
        }
    }
}
