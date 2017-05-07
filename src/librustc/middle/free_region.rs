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

use hir::def_id::DefId;
use middle::region::RegionMaps;
use ty::{self, Lift, TyCtxt, Region};
use ty::wf::ImpliedBound;
use rustc_data_structures::transitive_relation::TransitiveRelation;

/// Combines a `RegionMaps` (which governs relationships between
/// scopes) and a `FreeRegionMap` (which governs relationships between
/// free regions) to yield a complete relation between concrete
/// regions.
///
/// This stuff is a bit convoluted and should be refactored, but as we
/// move to NLL it'll all go away anyhow.
pub struct RegionRelations<'a, 'gcx: 'tcx, 'tcx: 'a> {
    pub tcx: TyCtxt<'a, 'gcx, 'tcx>,

    /// context used to fetch the region maps
    pub context: DefId,

    /// region maps for the given context
    pub region_maps: &'a RegionMaps<'tcx>,

    /// free-region relationships
    pub free_regions: &'a FreeRegionMap<'tcx>,
}

impl<'a, 'gcx, 'tcx> RegionRelations<'a, 'gcx, 'tcx> {
    pub fn new(
        tcx: TyCtxt<'a, 'gcx, 'tcx>,
        context: DefId,
        region_maps: &'a RegionMaps<'tcx>,
        free_regions: &'a FreeRegionMap<'tcx>,
    ) -> Self {
        Self {
            tcx,
            context,
            region_maps,
            free_regions,
        }
    }

    /// Determines whether one region is a subregion of another.  This is intended to run *after
    /// inference* and sadly the logic is somewhat duplicated with the code in infer.rs.
    pub fn is_subregion_of(&self,
                           sub_region: ty::Region<'tcx>,
                           super_region: ty::Region<'tcx>)
                           -> bool {
        let result = sub_region == super_region || {
            match (sub_region, super_region) {
                (&ty::ReEmpty, _) |
                (_, &ty::ReStatic) =>
                    true,

                (&ty::ReScope(sub_scope), &ty::ReScope(super_scope)) =>
                    self.region_maps.is_subscope_of(sub_scope, super_scope),

                (&ty::ReScope(sub_scope), &ty::ReFree(ref fr)) => {
                    let fr_scope = self.region_maps.free_extent(self.tcx, fr);
                    self.region_maps.is_subscope_of(sub_scope, fr_scope) ||
                        self.is_static(super_region)
                }

                (&ty::ReFree(_), &ty::ReFree(_)) =>
                    self.free_regions.relation.contains(&sub_region, &super_region) ||
                        self.is_static(super_region),

                (&ty::ReStatic, &ty::ReFree(_)) =>
                    self.is_static(super_region),

                _ =>
                    false,
            }
        };
        debug!("is_subregion_of(sub_region={:?}, super_region={:?}) = {:?}",
               sub_region, super_region, result);
        result
    }

    /// Determines whether this free-region is required to be 'static
    fn is_static(&self, super_region: ty::Region<'tcx>) -> bool {
        debug!("is_static(super_region={:?})", super_region);
        match *super_region {
            ty::ReStatic => true,
            ty::ReFree(_) => {
                let re_static = self.tcx.mk_region(ty::ReStatic);
                self.free_regions.relation.contains(&re_static, &super_region)
            }
            _ => bug!("only free regions should be given to `is_static`")
        }
    }

    pub fn lub_free_regions(&self,
                            r_a: Region<'tcx>,
                            r_b: Region<'tcx>)
                            -> Region<'tcx> {
        self.free_regions.lub_free_regions(self.tcx, r_a, r_b)
    }
}

#[derive(Clone, RustcEncodable, RustcDecodable)]
pub struct FreeRegionMap<'tcx> {
    // Stores the relation `a < b`, where `a` and `b` are regions.
    //
    // Invariant: only free regions like `'x` or `'static` are stored
    // in this relation, not scopes.
    relation: TransitiveRelation<Region<'tcx>>
}

impl<'tcx> FreeRegionMap<'tcx> {
    pub fn new() -> Self {
        FreeRegionMap { relation: TransitiveRelation::new() }
    }

    pub fn is_empty(&self) -> bool {
        self.relation.is_empty()
    }

    pub fn relate_free_regions_from_implied_bounds(&mut self,
                                                   implied_bounds: &[ImpliedBound<'tcx>])
    {
        debug!("relate_free_regions_from_implied_bounds()");
        for implied_bound in implied_bounds {
            debug!("implied bound: {:?}", implied_bound);
            match *implied_bound {
                ImpliedBound::RegionSubRegion(a @ &ty::ReFree(_), b @ &ty::ReFree(_)) |
                ImpliedBound::RegionSubRegion(a @ &ty::ReStatic, b @ &ty::ReFree(_)) => {
                    self.relate_regions(a, b);
                }
                ImpliedBound::RegionSubRegion(..) |
                ImpliedBound::RegionSubParam(..) |
                ImpliedBound::RegionSubProjection(..) => {
                }
            }
        }
    }

    pub fn relate_free_regions_from_predicates(&mut self,
                                               predicates: &[ty::Predicate<'tcx>]) {
        debug!("relate_free_regions_from_predicates(predicates={:?})", predicates);
        for predicate in predicates {
            match *predicate {
                ty::Predicate::Projection(..) |
                ty::Predicate::Trait(..) |
                ty::Predicate::Equate(..) |
                ty::Predicate::Subtype(..) |
                ty::Predicate::WellFormed(..) |
                ty::Predicate::ObjectSafe(..) |
                ty::Predicate::ClosureKind(..) |
                ty::Predicate::TypeOutlives(..) => {
                    // No region bounds here
                }
                ty::Predicate::RegionOutlives(ty::Binder(ty::OutlivesPredicate(r_a, r_b))) => {
                    match (r_a, r_b) {
                        // `'static: 'x` is not notable
                        (&ty::ReStatic, &ty::ReFree(_)) => {},

                        (&ty::ReFree(_), &ty::ReStatic) |
                        (&ty::ReFree(_), &ty::ReFree(_)) => {
                            // Record that `'a:'b`. Or, put another way, `'b <= 'a`.
                            self.relate_regions(r_b, r_a);
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

    fn relate_regions(&mut self, sub: Region<'tcx>, sup: Region<'tcx>) {
        assert!(match *sub { ty::ReFree(_) | ty::ReStatic => true, _ => false });
        assert!(match *sup { ty::ReFree(_) | ty::ReStatic => true, _ => false });
        self.relation.add(sub, sup)
    }

    pub fn lub_free_regions<'a, 'gcx>(&self,
                                      tcx: TyCtxt<'a, 'gcx, 'tcx>,
                                      r_a: Region<'tcx>,
                                      r_b: Region<'tcx>)
                                      -> Region<'tcx> {
        assert!(match *r_a { ty::ReFree(_) => true, _ => false });
        assert!(match *r_b { ty::ReFree(_) => true, _ => false });
        let result = if r_a == r_b { r_a } else {
            match self.relation.postdom_upper_bound(&r_a, &r_b) {
                None => tcx.mk_region(ty::ReStatic),
                Some(r) => *r,
            }
        };
        debug!("lub_free_regions(r_a={:?}, r_b={:?}) = {:?}", r_a, r_b, result);
        result
    }
}

impl_stable_hash_for!(struct FreeRegionMap<'tcx> {
    relation
});

impl<'a, 'tcx> Lift<'tcx> for FreeRegionMap<'a> {
    type Lifted = FreeRegionMap<'tcx>;
    fn lift_to_tcx<'b, 'gcx>(&self, tcx: TyCtxt<'b, 'gcx, 'tcx>) -> Option<FreeRegionMap<'tcx>> {
        self.relation.maybe_map(|&fr| fr.lift_to_tcx(tcx))
                     .map(|relation| FreeRegionMap { relation })
    }
}
