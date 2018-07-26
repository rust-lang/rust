// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
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
use borrow_check::nll::universal_regions::UniversalRegions;
use borrow_check::nll::type_check::constraint_conversion;
use borrow_check::nll::type_check::{Locations, MirTypeckRegionConstraints};
use rustc::hir::def_id::DefId;
use rustc::infer::region_constraints::GenericKind;
use rustc::infer::InferCtxt;
use rustc::traits::query::outlives_bounds::{self, OutlivesBound};
use rustc::traits::query::type_op::{self, TypeOp};
use rustc::ty::{self, RegionVid, Ty};
use rustc_data_structures::transitive_relation::TransitiveRelation;
use std::rc::Rc;
use syntax::ast;

#[derive(Debug)]
crate struct UniversalRegionRelations<'tcx> {
    universal_regions: Rc<UniversalRegions<'tcx>>,

    /// Each RBP `('a, GK)` indicates that `GK: 'a` can be assumed to
    /// be true. These encode relationships like `T: 'a` that are
    /// added via implicit bounds.
    ///
    /// Each region here is guaranteed to be a key in the `indices`
    /// map.  We use the "original" regions (i.e., the keys from the
    /// map, and not the values) because the code in
    /// `process_registered_region_obligations` has some special-cased
    /// logic expecting to see (e.g.) `ReStatic`, and if we supplied
    /// our special inference variable there, we would mess that up.
    crate region_bound_pairs: Vec<(ty::Region<'tcx>, GenericKind<'tcx>)>,

    /// Stores the outlives relations that are known to hold from the
    /// implied bounds, in-scope where clauses, and that sort of
    /// thing.
    outlives: TransitiveRelation<RegionVid>,

    /// This is the `<=` relation; that is, if `a: b`, then `b <= a`,
    /// and we store that here. This is useful when figuring out how
    /// to express some local region in terms of external regions our
    /// caller will understand.
    inverse_outlives: TransitiveRelation<RegionVid>,
}

impl UniversalRegionRelations<'tcx> {
    crate fn create(
        infcx: &InferCtxt<'_, '_, 'tcx>,
        mir_def_id: DefId,
        param_env: ty::ParamEnv<'tcx>,
        location_table: &LocationTable,
        implicit_region_bound: Option<ty::Region<'tcx>>,
        universal_regions: &Rc<UniversalRegions<'tcx>>,
        constraints: &mut MirTypeckRegionConstraints<'tcx>,
        all_facts: &mut Option<AllFacts>,
    ) -> Self {
        let mir_node_id = infcx.tcx.hir.as_local_node_id(mir_def_id).unwrap();
        UniversalRegionRelationsBuilder {
            infcx,
            mir_def_id,
            mir_node_id,
            param_env,
            implicit_region_bound,
            constraints,
            location_table,
            all_facts,
            universal_regions: universal_regions.clone(),
            relations: UniversalRegionRelations {
                universal_regions: universal_regions.clone(),
                region_bound_pairs: Vec::new(),
                outlives: TransitiveRelation::new(),
                inverse_outlives: TransitiveRelation::new(),
            },
        }.create()
    }

    /// Records in the `outlives_relation` (and
    /// `inverse_outlives_relation`) that `fr_a: fr_b`. Invoked by the
    /// builder below.
    fn relate_universal_regions(&mut self, fr_a: RegionVid, fr_b: RegionVid) {
        debug!(
            "relate_universal_regions: fr_a={:?} outlives fr_b={:?}",
            fr_a, fr_b
        );
        self.outlives.add(fr_a, fr_b);
        self.inverse_outlives.add(fr_b, fr_a);
    }
}

struct UniversalRegionRelationsBuilder<'this, 'gcx: 'tcx, 'tcx: 'this> {
    infcx: &'this InferCtxt<'this, 'gcx, 'tcx>,
    mir_def_id: DefId,
    mir_node_id: ast::NodeId,
    param_env: ty::ParamEnv<'tcx>,
    location_table: &'this LocationTable,
    universal_regions: Rc<UniversalRegions<'tcx>>,
    relations: UniversalRegionRelations<'tcx>,
    implicit_region_bound: Option<ty::Region<'tcx>>,
    constraints: &'this mut MirTypeckRegionConstraints<'tcx>,
    all_facts: &'this mut Option<AllFacts>,
}

impl UniversalRegionRelationsBuilder<'cx, 'gcx, 'tcx> {
    crate fn create(mut self) -> UniversalRegionRelations<'tcx> {
        let unnormalized_input_output_tys = self
            .universal_regions
            .unnormalized_input_tys
            .iter()
            .cloned()
            .chain(Some(self.universal_regions.unnormalized_output_ty));

        // For each of the input/output types:
        // - Normalize the type. This will create some region
        //   constraints, which we buffer up because we are
        //   not ready to process them yet.
        // - Then compute the implied bounds. This will adjust
        //   the `relations.region_bound_pairs` and so forth.
        // - After this is done, we'll process the constraints, once
        //   the `relations` is built.
        let constraint_sets: Vec<_> = unnormalized_input_output_tys
            .flat_map(|ty| {
                debug!("build: input_or_output={:?}", ty);
                let (ty, constraints) = self
                    .param_env
                    .and(type_op::normalize::Normalize::new(ty))
                    .fully_perform(self.infcx)
                    .unwrap_or_else(|_| bug!("failed to normalize {:?}", ty));
                self.add_implied_bounds(ty);
                constraints
            })
            .collect();

        // Insert the facts we know from the predicates. Why? Why not.
        let param_env = self.param_env;
        self.add_outlives_bounds(outlives_bounds::explicit_outlives_bounds(param_env));

        // Finally:
        // - outlives is reflexive, so `'r: 'r` for every region `'r`
        // - `'static: 'r` for every region `'r`
        // - `'r: 'fn_body` for every (other) universally quantified
        //   region `'r`, all of which are provided by our caller
        let fr_static = self.universal_regions.fr_static;
        let fr_fn_body = self.universal_regions.fr_fn_body;
        for fr in self.universal_regions.universal_regions() {
            debug!(
                "build: relating free region {:?} to itself and to 'static",
                fr
            );
            self.relations.relate_universal_regions(fr, fr);
            self.relations.relate_universal_regions(fr_static, fr);
            self.relations.relate_universal_regions(fr, fr_fn_body);
        }

        for data in constraint_sets {
            constraint_conversion::ConstraintConversion::new(
                self.infcx.tcx,
                &self.universal_regions,
                &self.location_table,
                &self.relations.region_bound_pairs,
                self.implicit_region_bound,
                self.param_env,
                Locations::All,
                &mut self.constraints.outlives_constraints,
                &mut self.constraints.type_tests,
                &mut self.all_facts,
            ).convert_all(&data);
        }

        self.relations
    }

    /// Update the type of a single local, which should represent
    /// either the return type of the MIR or one of its arguments. At
    /// the same time, compute and add any implied bounds that come
    /// from this local.
    fn add_implied_bounds(&mut self, ty: Ty<'tcx>) {
        debug!("add_implied_bounds(ty={:?})", ty);
        let span = self.infcx.tcx.def_span(self.mir_def_id);
        let bounds = self
            .infcx
            .implied_outlives_bounds(self.param_env, self.mir_node_id, ty, span);
        self.add_outlives_bounds(bounds);
    }

    /// Registers the `OutlivesBound` items from `outlives_bounds` in
    /// the outlives relation as well as the region-bound pairs
    /// listing.
    fn add_outlives_bounds<I>(&mut self, outlives_bounds: I)
    where
        I: IntoIterator<Item = OutlivesBound<'tcx>>,
    {
        for outlives_bound in outlives_bounds {
            debug!("add_outlives_bounds(bound={:?})", outlives_bound);

            match outlives_bound {
                OutlivesBound::RegionSubRegion(r1, r2) => {
                    // The bound says that `r1 <= r2`; we store `r2: r1`.
                    let r1 = self.universal_regions.to_region_vid(r1);
                    let r2 = self.universal_regions.to_region_vid(r2);
                    self.relations.relate_universal_regions(r2, r1);
                }

                OutlivesBound::RegionSubParam(r_a, param_b) => {
                    self.relations
                        .region_bound_pairs
                        .push((r_a, GenericKind::Param(param_b)));
                }

                OutlivesBound::RegionSubProjection(r_a, projection_b) => {
                    self.relations
                        .region_bound_pairs
                        .push((r_a, GenericKind::Projection(projection_b)));
                }
            }
        }
    }
}
