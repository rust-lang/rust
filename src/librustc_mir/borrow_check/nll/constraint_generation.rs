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

    // Add the reborrow constraint at `location` so that `borrowed_place`
    // is valid for `borrow_region`.
    fn add_reborrow_constraint(
        &mut self,
        location: Location,
        borrow_region: ty::Region<'tcx>,
        borrowed_place: &Place<'tcx>,
    ) {
        let mut borrowed_place = borrowed_place;

        debug!("add_reborrow_constraint({:?}, {:?}, {:?})",
               location, borrow_region, borrowed_place);
        while let Projection(box PlaceProjection { base, elem }) = borrowed_place {
            debug!("add_reborrow_constraint - iteration {:?}", borrowed_place);

            match *elem {
                ProjectionElem::Deref => {
                    let tcx = self.infcx.tcx;
                    let base_ty = base.ty(self.mir, tcx).to_ty(tcx);

                    debug!("add_reborrow_constraint - base_ty = {:?}", base_ty);
                    match base_ty.sty {
                        ty::TyRef(ref_region, ty::TypeAndMut { ty: _, mutbl }) => {
                            let span = self.mir.source_info(location).span;
                            self.regioncx.add_outlives(
                                span,
                                ref_region.to_region_vid(),
                                borrow_region.to_region_vid(),
                                location.successor_within_block(),
                            );

                            match mutbl {
                                hir::Mutability::MutImmutable => {
                                    // Immutable reference. We don't need the base
                                    // to be valid for the entire lifetime of
                                    // the borrow.
                                    break
                                }
                                hir::Mutability::MutMutable => {
                                    // Mutable reference. We *do* need the base
                                    // to be valid, because after the base becomes
                                    // invalid, someone else can use our mutable deref.

                                    // This is in order to make the following function
                                    // illegal:
                                    // ```
                                    // fn unsafe_deref<'a, 'b>(x: &'a &'b mut T) -> &'b mut T {
                                    //     &mut *x
                                    // }
                                    // ```
                                    //
                                    // As otherwise you could clone `&mut T` using the
                                    // following function:
                                    // ```
                                    // fn bad(x: &mut T) -> (&mut T, &mut T) {
                                    //     let my_clone = unsafe_deref(&'a x);
                                    //     ENDREGION 'a;
                                    //     (my_clone, x)
                                    // }
                                    // ```
                                }
                            }
                        }
                        ty::TyRawPtr(..) => {
                            // deref of raw pointer, guaranteed to be valid
                            break
                        }
                        ty::TyAdt(def, _) if def.is_box() => {
                            // deref of `Box`, need the base to be valid - propagate
                        }
                        _ => bug!("unexpected deref ty {:?} in {:?}", base_ty, borrowed_place)
                    }
                }
                ProjectionElem::Field(..) |
                ProjectionElem::Downcast(..) |
                ProjectionElem::Index(..) |
                ProjectionElem::ConstantIndex { .. } |
                ProjectionElem::Subslice { .. } => {
                    // other field access
                }
            }

            // The "propagate" case. We need to check that our base is valid
            // for the borrow's lifetime.
            borrowed_place = base;
        }
    }
}
