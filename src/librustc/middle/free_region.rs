// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This file defines

use middle::implicator::Implication;
use middle::ty::{self, FreeRegion};
use util::common::can_reach;
use util::nodemap::{FnvHashMap, FnvHashSet};

#[derive(Clone)]
pub struct FreeRegionMap {
    /// `map` maps from a free region `a` to a list of
    /// free regions `bs` such that `a <= b for all b in bs`
    map: FnvHashMap<FreeRegion, Vec<FreeRegion>>,
    /// regions that are required to outlive (and therefore be
    /// equal to) 'static.
    statics: FnvHashSet<FreeRegion>
}

impl FreeRegionMap {
    pub fn new() -> FreeRegionMap {
        FreeRegionMap { map: FnvHashMap(), statics: FnvHashSet() }
    }

    pub fn relate_free_regions_from_implications<'tcx>(&mut self,
                                                       implications: &[Implication<'tcx>])
    {
        for implication in implications {
            debug!("implication: {:?}", implication);
            match *implication {
                Implication::RegionSubRegion(_, ty::ReFree(free_a), ty::ReFree(free_b)) => {
                    self.relate_free_regions(free_a, free_b);
                }
                Implication::RegionSubRegion(..) |
                Implication::RegionSubGeneric(..) |
                Implication::Predicate(..) => {
                }
            }
        }
    }

    pub fn relate_free_regions_from_predicates<'tcx>(&mut self,
                                                     tcx: &ty::ctxt<'tcx>,
                                                     predicates: &[ty::Predicate<'tcx>]) {
        debug!("relate_free_regions_from_predicates(predicates={:?})", predicates);
        for predicate in predicates {
            match *predicate {
                ty::Predicate::Projection(..) |
                ty::Predicate::Trait(..) |
                ty::Predicate::Equate(..) |
                ty::Predicate::TypeOutlives(..) => {
                    // No region bounds here
                }
                ty::Predicate::RegionOutlives(ty::Binder(ty::OutlivesPredicate(r_a, r_b))) => {
                    match (r_a, r_b) {
                        (ty::ReStatic, ty::ReFree(_)) => {},
                        (ty::ReFree(fr_a), ty::ReStatic) => self.relate_to_static(fr_a),
                        (ty::ReFree(fr_a), ty::ReFree(fr_b)) => {
                            // Record that `'a:'b`. Or, put another way, `'b <= 'a`.
                            self.relate_free_regions(fr_b, fr_a);
                        }
                        _ => {
                            // All named regions are instantiated with free regions.
                            tcx.sess.bug(
                                &format!("record_region_bounds: non free region: {:?} / {:?}",
                                         r_a,
                                         r_b));
                        }
                    }
                }
            }
        }
    }

    fn relate_to_static(&mut self, sup: FreeRegion) {
        self.statics.insert(sup);
    }

    fn relate_free_regions(&mut self, sub: FreeRegion, sup: FreeRegion) {
       let mut sups = self.map.entry(sub).or_insert(Vec::new());
        if !sups.contains(&sup) {
            sups.push(sup);
        }
    }

    /// Determines whether two free regions have a subregion relationship
    /// by walking the graph encoded in `map`.  Note that
    /// it is possible that `sub != sup` and `sub <= sup` and `sup <= sub`
    /// (that is, the user can give two different names to the same lifetime).
    pub fn sub_free_region(&self, sub: FreeRegion, sup: FreeRegion) -> bool {
        can_reach(&self.map, sub, sup) || self.is_static(&sup)
    }

    /// Determines whether one region is a subregion of another.  This is intended to run *after
    /// inference* and sadly the logic is somewhat duplicated with the code in infer.rs.
    pub fn is_subregion_of(&self,
                           tcx: &ty::ctxt,
                           sub_region: ty::Region,
                           super_region: ty::Region)
                           -> bool {
        debug!("is_subregion_of(sub_region={:?}, super_region={:?})",
               sub_region, super_region);

        sub_region == super_region || {
            match (sub_region, super_region) {
                (ty::ReEmpty, _) |
                (_, ty::ReStatic) =>
                    true,

                (ty::ReScope(sub_scope), ty::ReScope(super_scope)) =>
                    tcx.region_maps.is_subscope_of(sub_scope, super_scope),

                (ty::ReScope(sub_scope), ty::ReFree(ref fr)) =>
                    tcx.region_maps.is_subscope_of(sub_scope, fr.scope.to_code_extent()),

                (ty::ReFree(sub_fr), ty::ReFree(super_fr)) =>
                    self.sub_free_region(sub_fr, super_fr),

                (ty::ReStatic, ty::ReFree(ref sup_fr)) => self.is_static(sup_fr),

                _ =>
                    false,
            }
        }
    }

    /// Determines whether this free-region is required to be 'static
    pub fn is_static(&self, super_region: &ty::FreeRegion) -> bool {
        debug!("is_static(super_region={:?})", super_region);
        self.statics.iter().any(|s| can_reach(&self.map, *s, *super_region))
    }
}
