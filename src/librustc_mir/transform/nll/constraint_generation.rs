// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::mir::{Location, Mir};
use rustc::mir::transform::MirSource;
use rustc::infer::InferCtxt;
use rustc::traits::{self, ObligationCause};
use rustc::ty::{self, Ty};
use rustc::ty::fold::TypeFoldable;
use rustc::util::common::ErrorReported;
use rustc_data_structures::fx::FxHashSet;
use syntax::codemap::DUMMY_SP;

use super::LivenessResults;
use super::ToRegionIndex;
use super::region_infer::RegionInferenceContext;

pub(super) fn generate_constraints<'a, 'gcx, 'tcx>(
    infcx: &InferCtxt<'a, 'gcx, 'tcx>,
    regioncx: &mut RegionInferenceContext,
    mir: &Mir<'tcx>,
    mir_source: MirSource,
    liveness: &LivenessResults,
) {
    ConstraintGeneration {
        infcx,
        regioncx,
        mir,
        liveness,
        mir_source,
    }.add_constraints();
}

struct ConstraintGeneration<'constrain, 'gcx: 'tcx, 'tcx: 'constrain> {
    infcx: &'constrain InferCtxt<'constrain, 'gcx, 'tcx>,
    regioncx: &'constrain mut RegionInferenceContext,
    mir: &'constrain Mir<'tcx>,
    liveness: &'constrain LivenessResults,
    mir_source: MirSource,
}

impl<'constrain, 'gcx, 'tcx> ConstraintGeneration<'constrain, 'gcx, 'tcx> {
    fn add_constraints(&mut self) {
        // To start, add the liveness constraints.
        self.add_liveness_constraints();
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

            self.liveness
                .drop
                .simulate_block(self.mir, bb, |location, live_locals| {
                    for live_local in live_locals.iter() {
                        let live_local_ty = self.mir.local_decls[live_local].ty;
                        self.add_drop_live_constraint(live_local_ty, location);
                    }
                });
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
                let vid = live_region.to_region_index();
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
            let def_id = tcx.hir.local_def_id(self.mir_source.item_id());
            let param_env = self.infcx.tcx.param_env(def_id);
            for ty in dtorck_types {
                // FIXME -- I think that this may disregard some region obligations
                // or something. Do we care? -nmatsakis
                let cause = ObligationCause::dummy();
                match traits::fully_normalize(self.infcx, cause, param_env, &ty) {
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
}
