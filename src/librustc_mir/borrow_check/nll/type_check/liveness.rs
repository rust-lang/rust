// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use borrow_check::nll::type_check::AtLocation;
use dataflow::move_paths::{HasMoveData, MoveData};
use dataflow::MaybeInitializedPlaces;
use dataflow::{FlowAtLocation, FlowsAtLocation};
use rustc::infer::canonical::QueryRegionConstraint;
use rustc::mir::Local;
use rustc::mir::{BasicBlock, Location, Mir};
use rustc::traits::query::dropck_outlives::DropckOutlivesResult;
use rustc::traits::query::type_op::outlives::DropckOutlives;
use rustc::traits::query::type_op::TypeOp;
use rustc::ty::{Ty, TypeFoldable};
use rustc_data_structures::fx::FxHashMap;
use rustc_data_structures::indexed_vec::Idx;
use std::rc::Rc;
use util::liveness::{LivenessResults, LiveVariableMap};

use super::TypeChecker;

/// Combines liveness analysis with initialization analysis to
/// determine which variables are live at which points, both due to
/// ordinary uses and drops. Returns a set of (ty, location) pairs
/// that indicate which types must be live at which point in the CFG.
/// This vector is consumed by `constraint_generation`.
///
/// NB. This computation requires normalization; therefore, it must be
/// performed before
pub(super) fn generate<'gcx, 'tcx, V: LiveVariableMap>(
    cx: &mut TypeChecker<'_, 'gcx, 'tcx>,
    mir: &Mir<'tcx>,
    liveness: &LivenessResults<V>,
    flow_inits: &mut FlowAtLocation<MaybeInitializedPlaces<'_, 'gcx, 'tcx>>,
    move_data: &MoveData<'tcx>,
) {
    let mut generator = TypeLivenessGenerator {
        cx,
        mir,
        liveness,
        flow_inits,
        move_data,
        drop_data: FxHashMap(),
    };

    for bb in mir.basic_blocks().indices() {
        generator.add_liveness_constraints(bb);
    }
}

struct TypeLivenessGenerator<'gen, 'typeck, 'flow, 'gcx, 'tcx, V: LiveVariableMap>
where
    'typeck: 'gen,
    'flow: 'gen,
    'tcx: 'typeck + 'flow,
    'gcx: 'tcx,
    V: 'gen,
{
    cx: &'gen mut TypeChecker<'typeck, 'gcx, 'tcx>,
    mir: &'gen Mir<'tcx>,
    liveness: &'gen LivenessResults<V>,
    flow_inits: &'gen mut FlowAtLocation<MaybeInitializedPlaces<'flow, 'gcx, 'tcx>>,
    move_data: &'gen MoveData<'tcx>,
    drop_data: FxHashMap<Ty<'tcx>, DropData<'tcx>>,
}

struct DropData<'tcx> {
    dropck_result: DropckOutlivesResult<'tcx>,
    region_constraint_data: Option<Rc<Vec<QueryRegionConstraint<'tcx>>>>,
}

impl<'gen, 'typeck, 'flow, 'gcx, 'tcx, V:LiveVariableMap> TypeLivenessGenerator<'gen, 'typeck, 'flow, 'gcx, 'tcx, V> {
    /// Liveness constraints:
    ///
    /// > If a variable V is live at point P, then all regions R in the type of V
    /// > must include the point P.
    fn add_liveness_constraints(&mut self, bb: BasicBlock) {
        debug!("add_liveness_constraints(bb={:?})", bb);

        self.liveness
            .regular
            .simulate_block(self.mir, bb, |location, live_locals| {
                for live_local in live_locals.iter() {
                    let live_local_ty = self.mir.local_decls[live_local].ty;
                    Self::push_type_live_constraint(&mut self.cx, live_local_ty, location);
                }
            });

        let mut all_live_locals: Vec<(Location, Vec<V::LiveVar>)> = vec![];
        self.liveness
            .drop
            .simulate_block(self.mir, bb, |location, live_locals| {
                all_live_locals.push((location, live_locals.iter().collect()));
            });
        debug!(
            "add_liveness_constraints: all_live_locals={:#?}",
            all_live_locals
        );

        let terminator_index = self.mir.basic_blocks()[bb].statements.len();
        self.flow_inits.reset_to_entry_of(bb);
        while let Some((location, live_locals)) = all_live_locals.pop() {
            for live_local in live_locals {
                debug!(
                    "add_liveness_constraints: location={:?} live_local={:?}",
                    location, live_local
                );

                if log_enabled!(::log::Level::Debug) {
                    self.flow_inits.each_state_bit(|mpi_init| {
                        debug!(
                            "add_liveness_constraints: location={:?} initialized={:?}",
                            location,
                            &self.flow_inits.operator().move_data().move_paths[mpi_init]
                        );
                    });
                }

                let mpi = self.move_data.rev_lookup.find_local(live_local);
                if let Some(initialized_child) = self.flow_inits.has_any_child_of(mpi) {
                    debug!(
                        "add_liveness_constraints: mpi={:?} has initialized child {:?}",
                        self.move_data.move_paths[mpi],
                        self.move_data.move_paths[initialized_child]
                    );

                    let live_local_ty = self.mir.local_decls[live_local].ty;
                    self.add_drop_live_constraint(live_local, live_local_ty, location);
                }
            }

            if location.statement_index == terminator_index {
                debug!(
                    "add_liveness_constraints: reconstruct_terminator_effect from {:#?}",
                    location
                );
                self.flow_inits.reconstruct_terminator_effect(location);
            } else {
                debug!(
                    "add_liveness_constraints: reconstruct_statement_effect from {:#?}",
                    location
                );
                self.flow_inits.reconstruct_statement_effect(location);
            }
            self.flow_inits.apply_local_effect(location);
        }
    }

    /// Some variable with type `live_ty` is "regular live" at
    /// `location` -- i.e., it may be used later. This means that all
    /// regions appearing in the type `live_ty` must be live at
    /// `location`.
    fn push_type_live_constraint<T>(
        cx: &mut TypeChecker<'_, 'gcx, 'tcx>,
        value: T,
        location: Location,
    ) where
        T: TypeFoldable<'tcx>,
    {
        debug!(
            "push_type_live_constraint(live_ty={:?}, location={:?})",
            value, location
        );

        cx.tcx().for_each_free_region(&value, |live_region| {
            if let Some(ref mut borrowck_context) = cx.borrowck_context {
                let region_vid = borrowck_context.universal_regions.to_region_vid(live_region);
                borrowck_context.constraints.liveness_constraints.add_element(region_vid, location);

                if let Some(all_facts) = borrowck_context.all_facts {
                    let start_index = borrowck_context.location_table.start_index(location);
                    all_facts.region_live_at.push((region_vid, start_index));

                    let mid_index = borrowck_context.location_table.mid_index(location);
                    all_facts.region_live_at.push((region_vid, mid_index));
                }
            }
        });
    }

    /// Some variable with type `live_ty` is "drop live" at `location`
    /// -- i.e., it may be dropped later. This means that *some* of
    /// the regions in its type must be live at `location`. The
    /// precise set will depend on the dropck constraints, and in
    /// particular this takes `#[may_dangle]` into account.
    fn add_drop_live_constraint(
        &mut self,
        dropped_local: Local,
        dropped_ty: Ty<'tcx>,
        location: Location,
    ) {
        debug!(
            "add_drop_live_constraint(dropped_local={:?}, dropped_ty={:?}, location={:?})",
            dropped_local, dropped_ty, location
        );

        let drop_data = self.drop_data.entry(dropped_ty).or_insert_with({
            let cx = &mut self.cx;
            move || Self::compute_drop_data(cx, dropped_ty)
        });

        if let Some(data) = &drop_data.region_constraint_data {
            self.cx.push_region_constraints(location.boring(), data);
        }

        drop_data.dropck_result.report_overflows(
            self.cx.infcx.tcx,
            self.mir.source_info(location).span,
            dropped_ty,
        );

        // All things in the `outlives` array may be touched by
        // the destructor and must be live at this point.
        for &kind in &drop_data.dropck_result.kinds {
            Self::push_type_live_constraint(&mut self.cx, kind, location);
        }
    }

    fn compute_drop_data(
        cx: &mut TypeChecker<'_, 'gcx, 'tcx>,
        dropped_ty: Ty<'tcx>,
    ) -> DropData<'tcx> {
        debug!("compute_drop_data(dropped_ty={:?})", dropped_ty,);

        let param_env = cx.param_env;
        let (dropck_result, region_constraint_data) = param_env
            .and(DropckOutlives::new(dropped_ty))
            .fully_perform(cx.infcx)
            .unwrap();

        DropData {
            dropck_result,
            region_constraint_data,
        }
    }
}
