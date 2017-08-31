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
use middle::region;
use ty::{self, Lift, TyCtxt, Region};
use rustc_data_structures::transitive_relation::TransitiveRelation;

/// Combines a `region::ScopeTree` (which governs relationships between
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
    pub region_scope_tree: &'a region::ScopeTree,

    /// free-region relationships
    pub free_regions: &'a FreeRegionMap<'tcx>,
}

impl<'a, 'gcx, 'tcx> RegionRelations<'a, 'gcx, 'tcx> {
    pub fn new(
        tcx: TyCtxt<'a, 'gcx, 'tcx>,
        context: DefId,
        region_scope_tree: &'a region::ScopeTree,
        free_regions: &'a FreeRegionMap<'tcx>,
    ) -> Self {
        Self {
            tcx,
            context,
            region_scope_tree,
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
                    self.region_scope_tree.is_subscope_of(sub_scope, super_scope),

                (&ty::ReScope(sub_scope), &ty::ReEarlyBound(ref br)) => {
                    let fr_scope = self.region_scope_tree.early_free_scope(self.tcx, br);
                    self.region_scope_tree.is_subscope_of(sub_scope, fr_scope)
                }

                (&ty::ReScope(sub_scope), &ty::ReFree(ref fr)) => {
                    let fr_scope = self.region_scope_tree.free_scope(self.tcx, fr);
                    self.region_scope_tree.is_subscope_of(sub_scope, fr_scope)
                }

                (&ty::ReEarlyBound(_), &ty::ReEarlyBound(_)) |
                (&ty::ReFree(_), &ty::ReEarlyBound(_)) |
                (&ty::ReEarlyBound(_), &ty::ReFree(_)) |
                (&ty::ReFree(_), &ty::ReFree(_)) =>
                    self.free_regions.relation.contains(&sub_region, &super_region),

                _ =>
                    false,
            }
        };
        let result = result || self.is_static(super_region);
        debug!("is_subregion_of(sub_region={:?}, super_region={:?}) = {:?}",
               sub_region, super_region, result);
        result
    }

    /// Determines whether this free-region is required to be 'static
    fn is_static(&self, super_region: ty::Region<'tcx>) -> bool {
        debug!("is_static(super_region={:?})", super_region);
        match *super_region {
            ty::ReStatic => true,
            ty::ReEarlyBound(_) | ty::ReFree(_) => {
                let re_static = self.tcx.mk_region(ty::ReStatic);
                self.free_regions.relation.contains(&re_static, &super_region)
            }
            _ => false
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
                    self.relate_regions(r_b, r_a);
                }
            }
        }
    }

    // Record that `'sup:'sub`. Or, put another way, `'sub <= 'sup`.
    // (with the exception that `'static: 'x` is not notable)
    pub fn relate_regions(&mut self, sub: Region<'tcx>, sup: Region<'tcx>) {
        if (is_free(sub) || *sub == ty::ReStatic) && is_free(sup) {
            self.relation.add(sub, sup)
        }
    }

    pub fn lub_free_regions<'a, 'gcx>(&self,
                                      tcx: TyCtxt<'a, 'gcx, 'tcx>,
                                      r_a: Region<'tcx>,
                                      r_b: Region<'tcx>)
                                      -> Region<'tcx> {
        assert!(is_free(r_a));
        assert!(is_free(r_b));
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

fn is_free(r: Region) -> bool {
    match *r {
        ty::ReEarlyBound(_) | ty::ReFree(_) => true,
        _ => false
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
