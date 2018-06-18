// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use borrow_check::location::LocationTable;
use borrow_check::nll::facts::AllFacts;
use rustc::infer::region_constraints::Constraint;
use rustc::infer::region_constraints::RegionConstraintData;
use rustc::infer::region_constraints::{Verify, VerifyBound};
use rustc::mir::Mir;
use rustc::ty;
use std::iter;
use syntax::codemap::Span;

use super::region_infer::{RegionInferenceContext, RegionTest, TypeTest};
use super::type_check::Locations;
use super::type_check::MirTypeckRegionConstraints;
use super::type_check::OutlivesSet;

/// When the MIR type-checker executes, it validates all the types in
/// the MIR, and in the process generates a set of constraints that
/// must hold regarding the regions in the MIR, along with locations
/// *where* they must hold. This code takes those constriants and adds
/// them into the NLL `RegionInferenceContext`.
pub(super) fn generate<'tcx>(
    regioncx: &mut RegionInferenceContext<'tcx>,
    all_facts: &mut Option<AllFacts>,
    location_table: &LocationTable,
    mir: &Mir<'tcx>,
    constraints: &MirTypeckRegionConstraints<'tcx>,
) {
    SubtypeConstraintGenerator {
        regioncx,
        location_table,
        mir,
    }.generate(constraints, all_facts);
}

struct SubtypeConstraintGenerator<'cx, 'tcx: 'cx> {
    regioncx: &'cx mut RegionInferenceContext<'tcx>,
    location_table: &'cx LocationTable,
    mir: &'cx Mir<'tcx>,
}

impl<'cx, 'tcx> SubtypeConstraintGenerator<'cx, 'tcx> {
    fn generate(
        &mut self,
        constraints: &MirTypeckRegionConstraints<'tcx>,
        all_facts: &mut Option<AllFacts>,
    ) {
        let MirTypeckRegionConstraints {
            liveness_set,
            outlives_sets,
        } = constraints;

        debug!(
            "generate(liveness_set={} items, outlives_sets={} items)",
            liveness_set.len(),
            outlives_sets.len()
        );

        for (region, location, cause) in liveness_set {
            debug!("generate: {:#?} is live at {:#?}", region, location);
            let region_vid = self.to_region_vid(region);
            self.regioncx.add_live_point(region_vid, *location, &cause);
        }

        if let Some(all_facts) = all_facts {
            all_facts
                .region_live_at
                .extend(liveness_set.into_iter().flat_map(|(region, location, _)| {
                    let r = self.to_region_vid(region);
                    let p1 = self.location_table.start_index(*location);
                    let p2 = self.location_table.mid_index(*location);
                    iter::once((r, p1)).chain(iter::once((r, p2)))
                }));
        }

        for OutlivesSet { locations, data } in outlives_sets {
            debug!("generate: constraints at: {:#?}", locations);
            let RegionConstraintData {
                constraints,
                verifys,
                givens,
            } = data;

            let span = locations.span(self.mir);

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
                self.regioncx.add_outlives(span, b_vid, a_vid);

                if let Some(all_facts) = all_facts {
                    locations.each_point(self.mir, |location| {
                        // In the new analysis, all outlives relations etc
                        // "take effect" at the mid point of the
                        // statement(s) that require them.
                        all_facts.outlives.push((
                            b_vid,
                            a_vid,
                            self.location_table.mid_index(location),
                        ));
                    });
                }
            }

            for verify in verifys {
                let type_test = self.verify_to_type_test(verify, span, locations);
                self.regioncx.add_type_test(type_test);
            }

            assert!(
                givens.is_empty(),
                "MIR type-checker does not use givens (thank goodness)"
            );
        }
    }

    fn verify_to_type_test(
        &self,
        verify: &Verify<'tcx>,
        span: Span,
        locations: &Locations,
    ) -> TypeTest<'tcx> {
        let generic_kind = verify.kind;

        let lower_bound = self.to_region_vid(verify.region);

        let test = self.verify_bound_to_region_test(&verify.bound);

        TypeTest {
            generic_kind,
            lower_bound,
            locations: *locations,
            span,
            test,
        }
    }

    fn verify_bound_to_region_test(&self, verify_bound: &VerifyBound<'tcx>) -> RegionTest {
        match verify_bound {
            VerifyBound::AnyRegion(regions) => RegionTest::IsOutlivedByAnyRegionIn(
                regions.iter().map(|r| self.to_region_vid(r)).collect(),
            ),

            VerifyBound::AllRegions(regions) => RegionTest::IsOutlivedByAllRegionsIn(
                regions.iter().map(|r| self.to_region_vid(r)).collect(),
            ),

            VerifyBound::AnyBound(bounds) => RegionTest::Any(
                bounds
                    .iter()
                    .map(|b| self.verify_bound_to_region_test(b))
                    .collect(),
            ),

            VerifyBound::AllBounds(bounds) => RegionTest::All(
                bounds
                    .iter()
                    .map(|b| self.verify_bound_to_region_test(b))
                    .collect(),
            ),
        }
    }

    fn to_region_vid(&self, r: ty::Region<'tcx>) -> ty::RegionVid {
        self.regioncx.to_region_vid(r)
    }
}
