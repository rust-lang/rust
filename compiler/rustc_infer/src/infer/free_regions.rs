//! This module handles the relationships between "free regions", i.e., lifetime parameters.
//! Ordinarily, free regions are unrelated to one another, but they can be related via implied
//! or explicit bounds. In that case, we track the bounds using the `TransitiveRelation` type,
//! and use that to decide when one free region outlives another, and so forth.

use rustc_data_structures::transitive_relation::TransitiveRelation;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::{self, Lift, Region, TyCtxt};

/// Combines a `FreeRegionMap` and a `TyCtxt`.
///
/// This stuff is a bit convoluted and should be refactored, but as we
/// transition to NLL, it'll all go away anyhow.
pub struct RegionRelations<'a, 'tcx> {
    pub tcx: TyCtxt<'tcx>,

    /// The context used for debug messages
    pub context: DefId,

    /// Free-region relationships.
    pub free_regions: &'a FreeRegionMap<'tcx>,
}

impl<'a, 'tcx> RegionRelations<'a, 'tcx> {
    pub fn new(tcx: TyCtxt<'tcx>, context: DefId, free_regions: &'a FreeRegionMap<'tcx>) -> Self {
        Self { tcx, context, free_regions }
    }

    pub fn lub_free_regions(&self, r_a: Region<'tcx>, r_b: Region<'tcx>) -> Region<'tcx> {
        self.free_regions.lub_free_regions(self.tcx, r_a, r_b)
    }
}

#[derive(Clone, Debug, Default)]
pub struct FreeRegionMap<'tcx> {
    // Stores the relation `a < b`, where `a` and `b` are regions.
    //
    // Invariant: only free regions like `'x` or `'static` are stored
    // in this relation, not scopes.
    relation: TransitiveRelation<Region<'tcx>>,
}

impl<'tcx> FreeRegionMap<'tcx> {
    pub fn elements(&self) -> impl Iterator<Item = &Region<'tcx>> {
        self.relation.elements()
    }

    pub fn is_empty(&self) -> bool {
        self.relation.is_empty()
    }

    // Record that `'sup:'sub`. Or, put another way, `'sub <= 'sup`.
    // (with the exception that `'static: 'x` is not notable)
    pub fn relate_regions(&mut self, sub: Region<'tcx>, sup: Region<'tcx>) {
        debug!("relate_regions(sub={:?}, sup={:?})", sub, sup);
        if self.is_free_or_static(sub) && self.is_free(sup) {
            self.relation.add(sub, sup)
        }
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
        assert!(self.is_free_or_static(r_a) && self.is_free_or_static(r_b));
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
        r_a == r_b || self.relation.contains(&r_a, &r_b)
    }

    /// True for free regions other than `'static`.
    pub fn is_free(&self, r: Region<'_>) -> bool {
        matches!(r, ty::ReEarlyBound(_) | ty::ReFree(_))
    }

    /// True if `r` is a free region or static of the sort that this
    /// free region map can be used with.
    pub fn is_free_or_static(&self, r: Region<'_>) -> bool {
        match *r {
            ty::ReStatic => true,
            _ => self.is_free(r),
        }
    }

    /// Computes the least-upper-bound of two free regions. In some
    /// cases, this is more conservative than necessary, in order to
    /// avoid making arbitrary choices. See
    /// `TransitiveRelation::postdom_upper_bound` for more details.
    pub fn lub_free_regions(
        &self,
        tcx: TyCtxt<'tcx>,
        r_a: Region<'tcx>,
        r_b: Region<'tcx>,
    ) -> Region<'tcx> {
        debug!("lub_free_regions(r_a={:?}, r_b={:?})", r_a, r_b);
        assert!(self.is_free(r_a));
        assert!(self.is_free(r_b));
        let result = if r_a == r_b {
            r_a
        } else {
            match self.relation.postdom_upper_bound(&r_a, &r_b) {
                None => tcx.lifetimes.re_static,
                Some(r) => *r,
            }
        };
        debug!("lub_free_regions(r_a={:?}, r_b={:?}) = {:?}", r_a, r_b, result);
        result
    }
}

impl<'a, 'tcx> Lift<'tcx> for FreeRegionMap<'a> {
    type Lifted = FreeRegionMap<'tcx>;
    fn lift_to_tcx(self, tcx: TyCtxt<'tcx>) -> Option<FreeRegionMap<'tcx>> {
        self.relation.maybe_map(|&fr| tcx.lift(fr)).map(|relation| FreeRegionMap { relation })
    }
}
