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
use borrow_check::nll::type_check::{Locations, LexicalRegionConstraintData};
use borrow_check::nll::universal_regions::UniversalRegions;
use rustc::infer::{self, RegionObligation, SubregionOrigin};
use rustc::infer::outlives::obligations::{TypeOutlives, TypeOutlivesDelegate};
use rustc::infer::region_constraints::{Constraint, GenericKind, VerifyBound};
use rustc::mir::{Location, Mir};
use rustc::ty::{self, TyCtxt};
use syntax::codemap::Span;

crate struct ConstraintConversion<'a, 'gcx: 'tcx, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'gcx, 'tcx>,
    mir: &'a Mir<'tcx>,
    universal_regions: &'a UniversalRegions<'tcx>,
    location_table: &'a LocationTable,
    region_bound_pairs: &'a [(ty::Region<'tcx>, GenericKind<'tcx>)],
    implicit_region_bound: Option<ty::Region<'tcx>>,
    param_env: ty::ParamEnv<'tcx>,
    locations: Locations,
    outlives_constraints: &'a mut Vec<OutlivesConstraint>,
    type_tests: &'a mut Vec<TypeTest<'tcx>>,
    all_facts: &'a mut Option<AllFacts>,
}

impl<'a, 'gcx, 'tcx> ConstraintConversion<'a, 'gcx, 'tcx> {
    crate fn new(
        tcx: TyCtxt<'a, 'gcx, 'tcx>,
        mir: &'a Mir<'tcx>,
        universal_regions: &'a UniversalRegions<'tcx>,
        location_table: &'a LocationTable,
        region_bound_pairs: &'a [(ty::Region<'tcx>, GenericKind<'tcx>)],
        implicit_region_bound: Option<ty::Region<'tcx>>,
        param_env: ty::ParamEnv<'tcx>,
        locations: Locations,
        outlives_constraints: &'a mut Vec<OutlivesConstraint>,
        type_tests: &'a mut Vec<TypeTest<'tcx>>,
        all_facts: &'a mut Option<AllFacts>,
    ) -> Self {
        Self {
            tcx,
            mir,
            universal_regions,
            location_table,
            region_bound_pairs,
            implicit_region_bound,
            param_env,
            locations,
            outlives_constraints,
            type_tests,
            all_facts,
        }
    }

    pub(super) fn convert(&mut self, data: &LexicalRegionConstraintData<'tcx>) {
        debug!("generate: constraints at: {:#?}", self.locations);
        let LexicalRegionConstraintData {
            constraints,
            region_obligations,
        } = data;

        for constraint in constraints {
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
            self.add_outlives(b_vid, a_vid);

            // In the new analysis, all outlives relations etc
            // "take effect" at the mid point of the statement
            // that requires them, so ignore the `at_location`.
            if let Some(all_facts) = &mut self.all_facts {
                if let Some(from_location) = self.locations.from_location() {
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

        let ConstraintConversion {
            tcx,
            region_bound_pairs,
            implicit_region_bound,
            param_env,
            ..
        } = *self;
        for r_o in region_obligations {
            let RegionObligation {
                sup_type,
                sub_region,
                cause,
            } = r_o;

            // we don't actually use this for anything.
            let origin = infer::RelateParamBound(cause.span, sup_type);

            TypeOutlives::new(
                &mut *self,
                tcx,
                region_bound_pairs,
                implicit_region_bound,
                param_env,
            ).type_must_outlive(origin, sup_type, sub_region);
        }
    }

    fn verify_to_type_test(
        &self,
        generic_kind: GenericKind<'tcx>,
        region: ty::Region<'tcx>,
        bound: VerifyBound<'tcx>,
    ) -> TypeTest<'tcx> {
        let lower_bound = self.to_region_vid(region);

        let point = self.locations.at_location().unwrap_or(Location::START);

        let test = self.verify_bound_to_region_test(&bound);

        TypeTest {
            generic_kind,
            lower_bound,
            point,
            span: self.span(),
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

    fn span(&self) -> Span {
        self
            .mir
            .source_info(self.locations.from_location().unwrap_or(Location::START))
            .span
    }

    fn add_outlives(
        &mut self,
        sup: ty::RegionVid,
        sub: ty::RegionVid,
    ) {
        let span = self.span();
        let point = self.locations.at_location().unwrap_or(Location::START);

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

impl<'a, 'b, 'gcx, 'tcx> TypeOutlivesDelegate<'tcx> for &'a mut ConstraintConversion<'b, 'gcx, 'tcx> {
    fn push_sub_region_constraint(
        &mut self,
        _origin: SubregionOrigin<'tcx>,
        a: ty::Region<'tcx>,
        b: ty::Region<'tcx>,
    ) {
        let b = self.universal_regions.to_region_vid(b);
        let a = self.universal_regions.to_region_vid(a);
        self.add_outlives(b, a);
    }

    fn push_verify(
        &mut self,
        _origin: SubregionOrigin<'tcx>,
        kind: GenericKind<'tcx>,
        a: ty::Region<'tcx>,
        bound: VerifyBound<'tcx>,
    ) {
        let type_test = self.verify_to_type_test(kind, a, bound);
        self.add_type_test(type_test);
    }
}
