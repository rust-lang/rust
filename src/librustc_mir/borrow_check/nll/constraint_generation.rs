// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use borrow_check::borrow_set::BorrowSet;
use borrow_check::location::LocationTable;
use borrow_check::nll::ToRegionVid;
use borrow_check::nll::facts::AllFacts;
use borrow_check::nll::region_infer::RegionInferenceContext;
use rustc::infer::InferCtxt;
use rustc::mir::visit::TyContext;
use rustc::mir::visit::Visitor;
use rustc::mir::{BasicBlock, BasicBlockData, Location, Mir, Place, Rvalue};
use rustc::mir::{Local, Statement, Terminator};
use rustc::ty::fold::TypeFoldable;
use rustc::ty::subst::Substs;
use rustc::ty::{self, CanonicalTy, ClosureSubsts, GeneratorSubsts};
use std::iter;

pub(super) fn generate_constraints<'cx, 'gcx, 'tcx>(
    infcx: &InferCtxt<'cx, 'gcx, 'tcx>,
    regioncx: &mut RegionInferenceContext<'tcx>,
    all_facts: &mut Option<AllFacts>,
    location_table: &LocationTable,
    mir: &Mir<'tcx>,
    borrow_set: &BorrowSet<'tcx>,
    liveness_set_from_typeck: &[(ty::Region<'tcx>, Location)],
) {
    let mut cg = ConstraintGeneration {
        borrow_set,
        infcx,
        regioncx,
        location_table,
        all_facts,
    };

    cg.add_region_liveness_constraints_from_type_check(liveness_set_from_typeck);

    for (bb, data) in mir.basic_blocks().iter_enumerated() {
        cg.visit_basic_block_data(bb, data);
    }
}

/// 'cg = the duration of the constraint generation process itself.
struct ConstraintGeneration<'cg, 'cx: 'cg, 'gcx: 'tcx, 'tcx: 'cx> {
    infcx: &'cg InferCtxt<'cx, 'gcx, 'tcx>,
    all_facts: &'cg mut Option<AllFacts>,
    location_table: &'cg LocationTable,
    regioncx: &'cg mut RegionInferenceContext<'tcx>,
    borrow_set: &'cg BorrowSet<'tcx>,
}

impl<'cg, 'cx, 'gcx, 'tcx> Visitor<'tcx> for ConstraintGeneration<'cg, 'cx, 'gcx, 'tcx> {
    fn visit_basic_block_data(&mut self, bb: BasicBlock, data: &BasicBlockData<'tcx>) {
        self.super_basic_block_data(bb, data);
    }

    /// We sometimes have `substs` within an rvalue, or within a
    /// call. Make them live at the location where they appear.
    fn visit_substs(&mut self, substs: &&'tcx Substs<'tcx>, location: Location) {
        self.add_regular_live_constraint(*substs, location);
        self.super_substs(substs);
    }

    /// We sometimes have `region` within an rvalue, or within a
    /// call. Make them live at the location where they appear.
    fn visit_region(&mut self, region: &ty::Region<'tcx>, location: Location) {
        self.add_regular_live_constraint(*region, location);
        self.super_region(region);
    }

    /// We sometimes have `ty` within an rvalue, or within a
    /// call. Make them live at the location where they appear.
    fn visit_ty(&mut self, ty: &ty::Ty<'tcx>, ty_context: TyContext) {
        match ty_context {
            TyContext::ReturnTy(source_info)
            | TyContext::YieldTy(source_info)
            | TyContext::LocalDecl { source_info, .. } => {
                span_bug!(
                    source_info.span,
                    "should not be visiting outside of the CFG: {:?}",
                    ty_context
                );
            }
            TyContext::Location(location) => {
                self.add_regular_live_constraint(*ty, location);
            }
        }

        self.super_ty(ty);
    }

    /// We sometimes have `generator_substs` within an rvalue, or within a
    /// call. Make them live at the location where they appear.
    fn visit_generator_substs(&mut self, substs: &GeneratorSubsts<'tcx>, location: Location) {
        self.add_regular_live_constraint(*substs, location);
        self.super_generator_substs(substs);
    }

    /// We sometimes have `closure_substs` within an rvalue, or within a
    /// call. Make them live at the location where they appear.
    fn visit_closure_substs(&mut self, substs: &ClosureSubsts<'tcx>, location: Location) {
        self.add_regular_live_constraint(*substs, location);
        self.super_closure_substs(substs);
    }

    fn visit_statement(
        &mut self,
        block: BasicBlock,
        statement: &Statement<'tcx>,
        location: Location,
    ) {
        if let Some(all_facts) = self.all_facts {
            all_facts.cfg_edge.push((
                self.location_table.start_index(location),
                self.location_table.mid_index(location),
            ));

            all_facts.cfg_edge.push((
                self.location_table.mid_index(location),
                self.location_table
                    .start_index(location.successor_within_block()),
            ));
        }

        self.super_statement(block, statement, location);
    }

    fn visit_assign(
        &mut self,
        block: BasicBlock,
        place: &Place<'tcx>,
        rvalue: &Rvalue<'tcx>,
        location: Location,
    ) {
        // When we see `X = ...`, then kill borrows of
        // `(*X).foo` and so forth.
        if let Some(all_facts) = self.all_facts {
            if let Place::Local(temp) = place {
                if let Some(borrow_indices) = self.borrow_set.local_map.get(temp) {
                    for &borrow_index in borrow_indices {
                        let location_index = self.location_table.mid_index(location);
                        all_facts.killed.push((borrow_index, location_index));
                    }
                }
            }
        }

        self.super_assign(block, place, rvalue, location);
    }

    fn visit_terminator(
        &mut self,
        block: BasicBlock,
        terminator: &Terminator<'tcx>,
        location: Location,
    ) {
        if let Some(all_facts) = self.all_facts {
            all_facts.cfg_edge.push((
                self.location_table.start_index(location),
                self.location_table.mid_index(location),
            ));

            for successor_block in terminator.successors() {
                all_facts.cfg_edge.push((
                    self.location_table.mid_index(location),
                    self.location_table
                        .start_index(successor_block.start_location()),
                ));
            }
        }

        self.super_terminator(block, terminator, location);
    }

    fn visit_user_assert_ty(
        &mut self,
        _c_ty: &CanonicalTy<'tcx>,
        _local: &Local,
        _location: Location,
    ) {
    }
}

impl<'cx, 'cg, 'gcx, 'tcx> ConstraintGeneration<'cx, 'cg, 'gcx, 'tcx> {
    /// The MIR type checker generates region liveness constraints
    /// that we also have to respect.
    fn add_region_liveness_constraints_from_type_check(
        &mut self,
        liveness_set: &[(ty::Region<'tcx>, Location)],
    ) {
        debug!(
            "add_region_liveness_constraints_from_type_check(liveness_set={} items)",
            liveness_set.len(),
        );

        let ConstraintGeneration {
            regioncx,
            location_table,
            all_facts,
            ..
        } = self;

        for (region, location) in liveness_set {
            debug!("generate: {:#?} is live at {:#?}", region, location);
            let region_vid = regioncx.to_region_vid(region);
            regioncx.add_live_point(region_vid, *location);
        }

        if let Some(all_facts) = all_facts {
            all_facts
                .region_live_at
                .extend(liveness_set.into_iter().flat_map(|(region, location)| {
                    let r = regioncx.to_region_vid(region);
                    let p1 = location_table.start_index(*location);
                    let p2 = location_table.mid_index(*location);
                    iter::once((r, p1)).chain(iter::once((r, p2)))
                }));
        }
    }

    /// Some variable with type `live_ty` is "regular live" at
    /// `location` -- i.e., it may be used later. This means that all
    /// regions appearing in the type `live_ty` must be live at
    /// `location`.
    fn add_regular_live_constraint<T>(&mut self, live_ty: T, location: Location)
    where
        T: TypeFoldable<'tcx>,
    {
        debug!(
            "add_regular_live_constraint(live_ty={:?}, location={:?})",
            live_ty, location
        );

        self.infcx
            .tcx
            .for_each_free_region(&live_ty, |live_region| {
                let vid = live_region.to_region_vid();
                self.regioncx.add_live_point(vid, location);
            });
    }
}
