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
use rustc::mir::{Local, Location, Place, Mir, Rvalue};
use rustc::mir::visit::Visitor;
use rustc::mir::Place::Projection;
use rustc::mir::{PlaceProjection, ProjectionElem};
use rustc::infer::InferCtxt;
use rustc::traits::{self, ObligationCause};
use rustc::ty::{self, Ty};
use rustc::ty::fold::TypeFoldable;
use rustc::util::common::ErrorReported;
use rustc_data_structures::fx::FxHashSet;
use syntax::codemap::DUMMY_SP;
use borrow_check::FlowInProgress;
use dataflow::MaybeInitializedLvals;
use dataflow::move_paths::{MoveData, HasMoveData};

use super::LivenessResults;
use super::ToRegionVid;
use super::region_infer::RegionInferenceContext;

pub(super) fn generate_constraints<'cx, 'gcx, 'tcx>(
    infcx: &InferCtxt<'cx, 'gcx, 'tcx>,
    regioncx: &mut RegionInferenceContext<'tcx>,
    mir: &Mir<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    liveness: &LivenessResults,
    flow_inits: &mut FlowInProgress<MaybeInitializedLvals<'cx, 'gcx, 'tcx>>,
    move_data: &MoveData<'tcx>,
) {
    ConstraintGeneration {
        infcx,
        regioncx,
        mir,
        liveness,
        param_env,
        flow_inits,
        move_data,
    }.add_constraints();
}

/// 'cg = the duration of the constraint generation process itself.
struct ConstraintGeneration<'cg, 'cx: 'cg, 'gcx: 'tcx, 'tcx: 'cx> {
    infcx: &'cg InferCtxt<'cx, 'gcx, 'tcx>,
    regioncx: &'cg mut RegionInferenceContext<'tcx>,
    mir: &'cg Mir<'tcx>,
    liveness: &'cg LivenessResults,
    param_env: ty::ParamEnv<'tcx>,
    flow_inits: &'cg mut FlowInProgress<MaybeInitializedLvals<'cx, 'gcx, 'tcx>>,
    move_data: &'cg MoveData<'tcx>,
}

impl<'cx, 'cg, 'gcx, 'tcx> ConstraintGeneration<'cx, 'cg, 'gcx, 'tcx> {
    fn add_constraints(&mut self) {
        self.add_liveness_constraints();
        self.add_borrow_constraints();
    }

    /// Liveness constraints:
    ///
    /// > If a variable V is live at point P, then all regions R in the type of V
    /// > must include the point P.
    fn add_liveness_constraints(&mut self) {
        debug!("add_liveness_constraints()");
        for bb in self.mir.basic_blocks().indices() {
            debug!("add_liveness_constraints: bb={:?}", bb);

            self.liveness
                .regular
                .simulate_block(self.mir, bb, |location, live_locals| {
                    for live_local in live_locals.iter() {
                        let live_local_ty = self.mir.local_decls[live_local].ty;
                        self.add_regular_live_constraint(live_local_ty, location);
                    }
                });

            let mut all_live_locals: Vec<(Location, Vec<Local>)> = vec![];
            self.liveness.drop.simulate_block(self.mir, bb, |location, live_locals| {
                all_live_locals.push((location, live_locals.iter().collect()));
            });
            debug!("add_liveness_constraints: all_live_locals={:#?}", all_live_locals);

            let terminator_index = self.mir.basic_blocks()[bb].statements.len();
            self.flow_inits.reset_to_entry_of(bb);
            while let Some((location, live_locals)) = all_live_locals.pop() {
                for live_local in live_locals {
                    debug!("add_liveness_constraints: location={:?} live_local={:?}", location,
                           live_local);

                    self.flow_inits.each_state_bit(|mpi_init| {
                        debug!("add_liveness_constraints: location={:?} initialized={:?}",
                               location,
                               &self.flow_inits
                                   .base_results
                                   .operator()
                                   .move_data()
                                   .move_paths[mpi_init]);
                    });

                    let mpi = self.move_data.rev_lookup.find_local(live_local);
                    if let Some(initialized_child) = self.flow_inits.has_any_child_of(mpi) {
                        debug!("add_liveness_constraints: mpi={:?} has initialized child {:?}",
                               self.move_data.move_paths[mpi],
                               self.move_data.move_paths[initialized_child]);

                        let live_local_ty = self.mir.local_decls[live_local].ty;
                        self.add_drop_live_constraint(live_local_ty, location);
                    }
                }

                if location.statement_index == terminator_index {
                    debug!("add_liveness_constraints: reconstruct_terminator_effect from {:#?}",
                           location);
                    self.flow_inits.reconstruct_terminator_effect(location);
                } else {
                    debug!("add_liveness_constraints: reconstruct_statement_effect from {:#?}",
                           location);
                    self.flow_inits.reconstruct_statement_effect(location);
                }
                self.flow_inits.apply_local_effect();
            }
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
            live_ty,
            location
        );

        self.infcx
            .tcx
            .for_each_free_region(&live_ty, |live_region| {
                let vid = live_region.to_region_vid();
                self.regioncx.add_live_point(vid, location);
            });
    }

    /// Some variable with type `live_ty` is "drop live" at `location`
    /// -- i.e., it may be dropped later. This means that *some* of
    /// the regions in its type must be live at `location`. The
    /// precise set will depend on the dropck constraints, and in
    /// particular this takes `#[may_dangle]` into account.
    fn add_drop_live_constraint(&mut self, dropped_ty: Ty<'tcx>, location: Location) {
        debug!(
            "add_drop_live_constraint(dropped_ty={:?}, location={:?})",
            dropped_ty,
            location
        );

        let tcx = self.infcx.tcx;
        let mut types = vec![(dropped_ty, 0)];
        let mut known = FxHashSet();
        while let Some((ty, depth)) = types.pop() {
            let span = DUMMY_SP; // FIXME
            let result = match tcx.dtorck_constraint_for_ty(span, dropped_ty, depth, ty) {
                Ok(result) => result,
                Err(ErrorReported) => {
                    continue;
                }
            };

            let ty::DtorckConstraint {
                outlives,
                dtorck_types,
            } = result;

            // All things in the `outlives` array may be touched by
            // the destructor and must be live at this point.
            for outlive in outlives {
                if let Some(ty) = outlive.as_type() {
                    self.add_regular_live_constraint(ty, location);
                } else if let Some(r) = outlive.as_region() {
                    self.add_regular_live_constraint(r, location);
                } else {
                    bug!()
                }
            }

            // However, there may also be some types that
            // `dtorck_constraint_for_ty` could not resolve (e.g.,
            // associated types and parameters). We need to normalize
            // associated types here and possibly recursively process.
            for ty in dtorck_types {
                let cause = ObligationCause::dummy();
                // We know that our original `dropped_ty` is well-formed,
                // so region obligations resulting from this normalization
                // should always hold.
                //
                // Therefore we ignore them instead of trying to match
                // them up with a location.
                let fulfillcx = traits::FulfillmentContext::new_ignoring_regions();
                match traits::fully_normalize_with_fulfillcx(
                    self.infcx, fulfillcx, cause, self.param_env, &ty
                ) {
                    Ok(ty) => match ty.sty {
                        ty::TyParam(..) | ty::TyProjection(..) | ty::TyAnon(..) => {
                            self.add_regular_live_constraint(ty, location);
                        }

                        _ => if known.insert(ty) {
                            types.push((ty, depth + 1));
                        },
                    },

                    Err(errors) => {
                        self.infcx.report_fulfillment_errors(&errors, None);
                    }
                }
            }
        }
    }

    fn add_borrow_constraints(&mut self) {
        self.visit_mir(self.mir);
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

                if let ty::TyRef(base_region, ty::TypeAndMut{ ty: _, mutbl }) = *base_sty {
                    match mutbl {
                        hir::Mutability::MutImmutable => { },

                        hir::Mutability::MutMutable => {
                            self.add_reborrow_constraint(location, borrow_region, base);
                        },
                    }

                    let span = self.mir.source_info(location).span;
                    self.regioncx.add_outlives(span,
                                               base_region.to_region_vid(),
                                               borrow_region.to_region_vid(),
                                               location.successor_within_block());
                }
            }
        }
    }
}

impl<'cg, 'cx, 'gcx, 'tcx> Visitor<'tcx> for ConstraintGeneration<'cg, 'cx, 'gcx, 'tcx> {
    fn visit_rvalue(&mut self,
                    rvalue: &Rvalue<'tcx>,
                    location: Location) {
        debug!("visit_rvalue(rvalue={:?}, location={:?})", rvalue, location);

        // Look for an rvalue like:
        //
        //     & L
        //
        // where L is the path that is borrowed. In that case, we have
        // to add the reborrow constraints (which don't fall out
        // naturally from the type-checker).
        if let Rvalue::Ref(region, _bk, ref borrowed_place) = *rvalue {
            self.add_reborrow_constraint(location, region, borrowed_place);
        }

        self.super_rvalue(rvalue, location);
    }
}
