// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::hir;
use rustc::mir::{BasicBlock, BasicBlockData, Location, Place, Mir, Rvalue};
use rustc::mir::visit::Visitor;
use rustc::mir::Place::Projection;
use rustc::mir::{PlaceProjection, ProjectionElem};
use rustc::mir::visit::TyContext;
use rustc::infer::InferCtxt;
use rustc::ty::{self, ClosureSubsts};
use rustc::ty::subst::Substs;
use rustc::ty::fold::TypeFoldable;

use super::ToRegionVid;
use super::region_infer::{RegionInferenceContext, Cause};

pub(super) fn generate_constraints<'cx, 'gcx, 'tcx>(
    infcx: &InferCtxt<'cx, 'gcx, 'tcx>,
    regioncx: &mut RegionInferenceContext<'tcx>,
    mir: &Mir<'tcx>,
) {
    let mut cg = ConstraintGeneration {
        infcx,
        regioncx,
        mir,
    };

    for (bb, data) in mir.basic_blocks().iter_enumerated() {
        cg.visit_basic_block_data(bb, data);
    }
}

/// 'cg = the duration of the constraint generation process itself.
struct ConstraintGeneration<'cg, 'cx: 'cg, 'gcx: 'tcx, 'tcx: 'cx> {
    infcx: &'cg InferCtxt<'cx, 'gcx, 'tcx>,
    regioncx: &'cg mut RegionInferenceContext<'tcx>,
    mir: &'cg Mir<'tcx>,
}

impl<'cg, 'cx, 'gcx, 'tcx> Visitor<'tcx> for ConstraintGeneration<'cg, 'cx, 'gcx, 'tcx> {
    fn visit_basic_block_data(&mut self, bb: BasicBlock, data: &BasicBlockData<'tcx>) {
        self.super_basic_block_data(bb, data);
    }

    /// We sometimes have `substs` within an rvalue, or within a
    /// call. Make them live at the location where they appear.
    fn visit_substs(&mut self, substs: &&'tcx Substs<'tcx>, location: Location) {
        self.add_regular_live_constraint(*substs, location, Cause::LiveOther(location));
        self.super_substs(substs);
    }

    /// We sometimes have `region` within an rvalue, or within a
    /// call. Make them live at the location where they appear.
    fn visit_region(&mut self, region: &ty::Region<'tcx>, location: Location) {
        self.add_regular_live_constraint(*region, location, Cause::LiveOther(location));
        self.super_region(region);
    }

    /// We sometimes have `ty` within an rvalue, or within a
    /// call. Make them live at the location where they appear.
    fn visit_ty(&mut self, ty: &ty::Ty<'tcx>, ty_context: TyContext) {
        match ty_context {
            TyContext::ReturnTy(source_info) |
            TyContext::LocalDecl { source_info, .. } => {
                span_bug!(source_info.span,
                          "should not be visiting outside of the CFG: {:?}",
                          ty_context);
            }
            TyContext::Location(location) => {
                self.add_regular_live_constraint(*ty, location, Cause::LiveOther(location));
            }
        }

        self.super_ty(ty);
    }

    /// We sometimes have `closure_substs` within an rvalue, or within a
    /// call. Make them live at the location where they appear.
    fn visit_closure_substs(&mut self, substs: &ClosureSubsts<'tcx>, location: Location) {
        self.add_regular_live_constraint(*substs, location, Cause::LiveOther(location));
        self.super_closure_substs(substs);
    }

    fn visit_rvalue(&mut self, rvalue: &Rvalue<'tcx>, location: Location) {
        debug!("visit_rvalue(rvalue={:?}, location={:?})", rvalue, location);

        // Look for an rvalue like:
        //
        //     & L
        //
        // where L is the path that is borrowed. In that case, we have
        // to add the reborrow constraints (which don't fall out
        // naturally from the type-checker).
        if let Rvalue::Ref(region, _bk, ref borrowed_lv) = *rvalue {
            self.add_reborrow_constraint(location, region, borrowed_lv);
        }

        self.super_rvalue(rvalue, location);
    }
}

impl<'cx, 'cg, 'gcx, 'tcx> ConstraintGeneration<'cx, 'cg, 'gcx, 'tcx> {
    /// Some variable with type `live_ty` is "regular live" at
    /// `location` -- i.e., it may be used later. This means that all
    /// regions appearing in the type `live_ty` must be live at
    /// `location`.
    fn add_regular_live_constraint<T>(&mut self, live_ty: T, location: Location, cause: Cause)
    where
        T: TypeFoldable<'tcx>,
    {
        debug!(
            "add_regular_live_constraint(live_ty={:?}, location={:?})",
            live_ty,
            location
        );

        self.infcx
            .tcx
            .for_each_free_region(&live_ty, |live_region| {
                let vid = live_region.to_region_vid();
                self.regioncx.add_live_point(vid, location, &cause);
            });
    }

    fn add_reborrow_constraint(
        &mut self,
        location: Location,
        borrow_region: ty::Region<'tcx>,
        borrowed_place: &Place<'tcx>,
    ) {
        if let Projection(ref proj) = *borrowed_place {
            let PlaceProjection { ref base, ref elem } = **proj;

            if let ProjectionElem::Deref = *elem {
                let tcx = self.infcx.tcx;
                let base_ty = base.ty(self.mir, tcx).to_ty(tcx);
                let base_sty = &base_ty.sty;

                if let ty::TyRef(base_region, ty::TypeAndMut { ty: _, mutbl }) = *base_sty {
                    match mutbl {
                        hir::Mutability::MutImmutable => {}

                        hir::Mutability::MutMutable => {
                            self.add_reborrow_constraint(location, borrow_region, base);
                        }
                    }

                    let span = self.mir.source_info(location).span;
                    self.regioncx.add_outlives(
                        span,
                        base_region.to_region_vid(),
                        borrow_region.to_region_vid(),
                        location.successor_within_block(),
                    );
                }
            }
        }
    }
}
