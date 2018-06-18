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
use borrow_check::nll::region_infer::{OutlivesConstraint, RegionTest, TypeTest};
use borrow_check::nll::type_check::Locations;
use borrow_check::nll::universal_regions::UniversalRegions;
use rustc::infer::region_constraints::Constraint;
use rustc::infer::region_constraints::RegionConstraintData;
use rustc::infer::region_constraints::{Verify, VerifyBound};
use rustc::mir::{Location, Mir};
use rustc::ty;
use syntax::codemap::Span;

crate struct ConstraintConversion<'a, 'tcx: 'a> {
    mir: &'a Mir<'tcx>,
    universal_regions: &'a UniversalRegions<'tcx>,
    location_table: &'a LocationTable,
    outlives_constraints: &'a mut Vec<OutlivesConstraint>,
    type_tests: &'a mut Vec<TypeTest<'tcx>>,
    all_facts: &'a mut Option<AllFacts>,

}

impl<'a, 'tcx> ConstraintConversion<'a, 'tcx> {
    crate fn new(
        mir: &'a Mir<'tcx>,
        universal_regions: &'a UniversalRegions<'tcx>,
        location_table: &'a LocationTable,
        outlives_constraints: &'a mut Vec<OutlivesConstraint>,
        type_tests: &'a mut Vec<TypeTest<'tcx>>,
        all_facts: &'a mut Option<AllFacts>,
    ) -> Self {
        Self {
            mir,
            universal_regions,
            location_table,
            outlives_constraints,
            type_tests,
            all_facts,
        }
    }

    crate fn convert(
        &mut self,
        locations: Locations,
        data: &RegionConstraintData<'tcx>,
    ) {
        debug!("generate: constraints at: {:#?}", locations);
        let RegionConstraintData {
            constraints,
            verifys,
            givens,
        } = data;

        let span = self
            .mir
            .source_info(locations.from_location().unwrap_or(Location::START))
            .span;

        let at_location = locations.at_location().unwrap_or(Location::START);

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
            self.add_outlives(span, b_vid, a_vid, at_location);

            // In the new analysis, all outlives relations etc
            // "take effect" at the mid point of the statement
            // that requires them, so ignore the `at_location`.
            if let Some(all_facts) = &mut self.all_facts {
                if let Some(from_location) = locations.from_location() {
                    all_facts.outlives.push((
                        b_vid,
                        a_vid,
                        self.location_table.mid_index(from_location),
                    ));
                } else {
                    for location in self.location_table.all_points() {
                        all_facts.outlives.push((b_vid, a_vid, location));
                    }
                }
            }
        }

        for verify in verifys {
            let type_test = self.verify_to_type_test(verify, span, locations);
            self.add_type_test(type_test);
        }

        assert!(
            givens.is_empty(),
            "MIR type-checker does not use givens (thank goodness)"
        );
    }

    fn verify_to_type_test(
        &self,
        verify: &Verify<'tcx>,
        span: Span,
        locations: Locations,
    ) -> TypeTest<'tcx> {
        let generic_kind = verify.kind;

        let lower_bound = self.to_region_vid(verify.region);

        let point = locations.at_location().unwrap_or(Location::START);

        let test = self.verify_bound_to_region_test(&verify.bound);

        TypeTest {
            generic_kind,
            lower_bound,
            point,
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
        self.universal_regions.to_region_vid(r)
    }

    fn add_outlives(
        &mut self,
        span: Span,
        sup: ty::RegionVid,
        sub: ty::RegionVid,
        point: Location,
    ) {
        self.outlives_constraints.push(OutlivesConstraint {
            span,
            sub,
            sup,
            point,
            next: None,
        });
    }

    fn add_type_test(&mut self, type_test: TypeTest<'tcx>) {
        self.type_tests.push(type_test);
    }
}
