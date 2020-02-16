//! This module handles the relationships between "free regions", i.e., lifetime parameters.
//! Ordinarily, free regions are unrelated to one another, but they can be related via implied
//! or explicit bounds. In that case, we track the bounds using the `TransitiveRelation` type,
//! and use that to decide when one free region outlives another, and so forth.

use crate::middle::region;
use crate::ty::free_region_map::FreeRegionMap;
use crate::ty::{Region, TyCtxt};
use rustc_hir::def_id::DefId;

/// Combines a `region::ScopeTree` (which governs relationships between
/// scopes) and a `FreeRegionMap` (which governs relationships between
/// free regions) to yield a complete relation between concrete
/// regions.
///
/// This stuff is a bit convoluted and should be refactored, but as we
/// transition to NLL, it'll all go away anyhow.
pub struct RegionRelations<'a, 'tcx> {
    pub tcx: TyCtxt<'tcx>,

    /// The context used to fetch the region maps.
    pub context: DefId,

    /// The region maps for the given context.
    pub region_scope_tree: &'a region::ScopeTree,

    /// Free-region relationships.
    pub free_regions: &'a FreeRegionMap<'tcx>,
}

impl<'a, 'tcx> RegionRelations<'a, 'tcx> {
    pub fn new(
        tcx: TyCtxt<'tcx>,
        context: DefId,
        region_scope_tree: &'a region::ScopeTree,
        free_regions: &'a FreeRegionMap<'tcx>,
    ) -> Self {
        Self { tcx, context, region_scope_tree, free_regions }
    }

    pub fn lub_free_regions(&self, r_a: Region<'tcx>, r_b: Region<'tcx>) -> Region<'tcx> {
        self.free_regions.lub_free_regions(self.tcx, r_a, r_b)
    }
}
