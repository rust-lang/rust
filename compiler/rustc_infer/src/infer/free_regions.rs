//! This module handles the relationships between "free regions", i.e., lifetime parameters.
//! Ordinarily, free regions are unrelated to one another, but they can be related via implied
//! or explicit bounds. In that case, we track the bounds using the `TransitiveRelation` type,
//! and use that to decide when one free region outlives another, and so forth.

use rustc_data_structures::transitive_relation::TransitiveRelation;
use rustc_middle::ty::{Region, TyCtxt};
use tracing::debug;

/// Combines a `FreeRegionMap` and a `TyCtxt`.
///
/// This stuff is a bit convoluted and should be refactored, but as we
/// transition to NLL, it'll all go away anyhow.
pub(crate) struct RegionRelations<'a, 'tcx> {
    pub tcx: TyCtxt<'tcx>,

    /// Free-region relationships.
    pub free_regions: &'a FreeRegionMap<'tcx>,
}

impl<'a, 'tcx> RegionRelations<'a, 'tcx> {
    pub(crate) fn new(tcx: TyCtxt<'tcx>, free_regions: &'a FreeRegionMap<'tcx>) -> Self {
        Self { tcx, free_regions }
    }

    pub(crate) fn lub_param_regions(&self, r_a: Region<'tcx>, r_b: Region<'tcx>) -> Region<'tcx> {
        self.free_regions.lub_param_regions(self.tcx, r_a, r_b)
    }
}

#[derive(Clone, Debug)]
pub struct FreeRegionMap<'tcx> {
    /// Stores the relation `a < b`, where `a` and `b` are regions.
    ///
    /// Invariant: only free regions like `'x` or `'static` are stored
    /// in this relation, not scopes.
    pub(crate) relation: TransitiveRelation<Region<'tcx>>,
}

impl<'tcx> FreeRegionMap<'tcx> {
    pub fn elements(&self) -> impl Iterator<Item = Region<'tcx>> {
        self.relation.elements().copied()
    }

    pub fn is_empty(&self) -> bool {
        self.relation.is_empty()
    }

    /// Tests whether `r_a <= r_b`.
    ///
    /// Both regions must meet `is_free_or_static`.
    ///
    /// Subtle: one tricky case that this code gets correct is as
    /// follows. If we know that `r_b: 'static`, then this function
    /// will return true, even though we don't know anything that
    /// directly relates `r_a` and `r_b`.
    pub fn sub_free_regions(
        &self,
        tcx: TyCtxt<'tcx>,
        r_a: Region<'tcx>,
        r_b: Region<'tcx>,
    ) -> bool {
        assert!(r_a.is_free() && r_b.is_free());
        let re_static = tcx.lifetimes.re_static;
        if self.check_relation(re_static, r_b) {
            // `'a <= 'static` is always true, and not stored in the
            // relation explicitly, so check if `'b` is `'static` (or
            // equivalent to it)
            true
        } else {
            self.check_relation(r_a, r_b)
        }
    }

    /// Check whether `r_a <= r_b` is found in the relation.
    fn check_relation(&self, r_a: Region<'tcx>, r_b: Region<'tcx>) -> bool {
        r_a == r_b || self.relation.contains(r_a, r_b)
    }

    /// Computes the least-upper-bound of two free regions. In some
    /// cases, this is more conservative than necessary, in order to
    /// avoid making arbitrary choices. See
    /// `TransitiveRelation::postdom_upper_bound` for more details.
    pub(crate) fn lub_param_regions(
        &self,
        tcx: TyCtxt<'tcx>,
        r_a: Region<'tcx>,
        r_b: Region<'tcx>,
    ) -> Region<'tcx> {
        debug!("lub_param_regions(r_a={:?}, r_b={:?})", r_a, r_b);
        assert!(r_a.is_param());
        assert!(r_b.is_param());
        let result = if r_a == r_b {
            r_a
        } else {
            match self.relation.postdom_upper_bound(r_a, r_b) {
                None => tcx.lifetimes.re_static,
                Some(r) => r,
            }
        };
        debug!("lub_param_regions(r_a={:?}, r_b={:?}) = {:?}", r_a, r_b, result);
        result
    }
}
