// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This file handles the relationships between free regions --
//! meaning lifetime parameters. Ordinarily, free regions are
//! unrelated to one another, but they can be related via implied or
//! explicit bounds.  In that case, we track the bounds using the
//! `TransitiveRelation` type and use that to decide when one free
//! region outlives another and so forth.

use ty::{self, TyCtxt, FreeRegion, Region};
use ty::wf::ImpliedBound;
use rustc_data_structures::transitive_relation::TransitiveRelation;

#[derive(Clone)]
pub struct FreeRegionMap {
    // Stores the relation `a < b`, where `a` and `b` are regions.
    relation: TransitiveRelation<Region>
}

impl FreeRegionMap {
    pub fn new() -> FreeRegionMap {
        FreeRegionMap { relation: TransitiveRelation::new() }
    }

    pub fn relate_free_regions_from_implied_bounds<'tcx>(&mut self,
                                                        implied_bounds: &[ImpliedBound<'tcx>])
    {
        debug!("relate_free_regions_from_implied_bounds()");
        for implied_bound in implied_bounds {
            debug!("implied bound: {:?}", implied_bound);
            match *implied_bound {
                ImpliedBound::RegionSubRegion(&ty::ReFree(free_a), &ty::ReFree(free_b)) => {
                    self.relate_free_regions(free_a, free_b);
                }
                ImpliedBound::RegionSubRegion(..) |
                ImpliedBound::RegionSubParam(..) |
                ImpliedBound::RegionSubProjection(..) => {
                }
            }
        }
    }

    pub fn relate_free_regions_from_predicates(&mut self,
                                               predicates: &[ty::Predicate]) {
        debug!("relate_free_regions_from_predicates(predicates={:?})", predicates);
        for predicate in predicates {
            match *predicate {
                ty::Predicate::Projection(..) |
                ty::Predicate::Trait(..) |
                ty::Predicate::Equate(..) |
                ty::Predicate::WellFormed(..) |
                ty::Predicate::ObjectSafe(..) |
                ty::Predicate::ClosureKind(..) |
                ty::Predicate::TypeOutlives(..) => {
                    // No region bounds here
                }
                ty::Predicate::RegionOutlives(ty::Binder(ty::OutlivesPredicate(r_a, r_b))) => {
                    match (r_a, r_b) {
                        (&ty::ReStatic, &ty::ReFree(_)) => {},
                        (&ty::ReFree(fr_a), &ty::ReStatic) => self.relate_to_static(fr_a),
                        (&ty::ReFree(fr_a), &ty::ReFree(fr_b)) => {
                            // Record that `'a:'b`. Or, put another way, `'b <= 'a`.
                            self.relate_free_regions(fr_b, fr_a);
                        }
                        _ => {
                            // All named regions are instantiated with free regions.
                            bug!("record_region_bounds: non free region: {:?} / {:?}",
                                 r_a,
                                 r_b);
                        }
                    }
                }
            }
        }
    }

    fn relate_to_static(&mut self, sup: FreeRegion) {
        self.relation.add(ty::ReStatic, ty::ReFree(sup));
    }

    fn relate_free_regions(&mut self, sub: FreeRegion, sup: FreeRegion) {
        self.relation.add(ty::ReFree(sub), ty::ReFree(sup))
    }

    /// Determines whether two free regions have a subregion relationship
    /// by walking the graph encoded in `map`.  Note that
    /// it is possible that `sub != sup` and `sub <= sup` and `sup <= sub`
    /// (that is, the user can give two different names to the same lifetime).
    pub fn sub_free_region(&self, sub: FreeRegion, sup: FreeRegion) -> bool {
        let result = sub == sup || {
            let sub = ty::ReFree(sub);
            let sup = ty::ReFree(sup);
            self.relation.contains(&sub, &sup) || self.relation.contains(&ty::ReStatic, &sup)
        };
        debug!("sub_free_region(sub={:?}, sup={:?}) = {:?}", sub, sup, result);
        result
    }

    pub fn lub_free_regions(&self, fr_a: FreeRegion, fr_b: FreeRegion) -> Region {
        let r_a = ty::ReFree(fr_a);
        let r_b = ty::ReFree(fr_b);
        let result = if fr_a == fr_b { r_a } else {
            match self.relation.postdom_upper_bound(&r_a, &r_b) {
                None => ty::ReStatic,
                Some(r) => *r,
            }
        };
        debug!("lub_free_regions(fr_a={:?}, fr_b={:?}) = {:?}", fr_a, fr_b, result);
        result
    }

    /// Determines whether one region is a subregion of another.  This is intended to run *after
    /// inference* and sadly the logic is somewhat duplicated with the code in infer.rs.
    pub fn is_subregion_of(&self,
                           tcx: TyCtxt,
                           sub_region: &ty::Region,
                           super_region: &ty::Region)
                           -> bool {
        let result = sub_region == super_region || {
            match (sub_region, super_region) {
                (&ty::ReEmpty, _) |
                (_, &ty::ReStatic) =>
                    true,

                (&ty::ReScope(sub_scope), &ty::ReScope(super_scope)) =>
                    tcx.region_maps.is_subscope_of(sub_scope, super_scope),

                (&ty::ReScope(sub_scope), &ty::ReFree(fr)) =>
                    tcx.region_maps.is_subscope_of(sub_scope, fr.scope) ||
                    self.is_static(fr),

                (&ty::ReFree(sub_fr), &ty::ReFree(super_fr)) =>
                    self.sub_free_region(sub_fr, super_fr),

                (&ty::ReStatic, &ty::ReFree(sup_fr)) =>
                    self.is_static(sup_fr),

                _ =>
                    false,
            }
        };
        debug!("is_subregion_of(sub_region={:?}, super_region={:?}) = {:?}",
               sub_region, super_region, result);
        result
    }

    /// Determines whether this free-region is required to be 'static
    pub fn is_static(&self, super_region: ty::FreeRegion) -> bool {
        debug!("is_static(super_region={:?})", super_region);
        self.relation.contains(&ty::ReStatic, &ty::ReFree(super_region))
    }
}

#[cfg(test)]
fn free_region(index: u32) -> FreeRegion {
    use middle::region::DUMMY_CODE_EXTENT;
    FreeRegion { scope: DUMMY_CODE_EXTENT,
                 bound_region: ty::BoundRegion::BrAnon(index) }
}

#[test]
fn lub() {
    // a very VERY basic test, but see the tests in
    // TransitiveRelation, which are much more thorough.
    let frs: Vec<_> = (0..3).map(|i| free_region(i)).collect();
    let mut map = FreeRegionMap::new();
    map.relate_free_regions(frs[0], frs[2]);
    map.relate_free_regions(frs[1], frs[2]);
    assert_eq!(map.lub_free_regions(frs[0], frs[1]), ty::ReFree(frs[2]));
}
