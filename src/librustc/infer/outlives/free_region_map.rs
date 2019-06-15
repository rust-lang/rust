use crate::ty::{self, Lift, TyCtxt, Region};
use rustc_data_structures::transitive_relation::TransitiveRelation;

#[derive(Clone, RustcEncodable, RustcDecodable, Debug, Default)]
pub struct FreeRegionMap<'tcx> {
    // Stores the relation `a < b`, where `a` and `b` are regions.
    //
    // Invariant: only free regions like `'x` or `'static` are stored
    // in this relation, not scopes.
    relation: TransitiveRelation<Region<'tcx>>
}

impl<'tcx> FreeRegionMap<'tcx> {
    pub fn elements(&self) -> impl Iterator<Item=&Region<'tcx>> {
        self.relation.elements()
    }

    pub fn is_empty(&self) -> bool {
        self.relation.is_empty()
    }

    // Record that `'sup:'sub`. Or, put another way, `'sub <= 'sup`.
    // (with the exception that `'static: 'x` is not notable)
    pub fn relate_regions(&mut self, sub: Region<'tcx>, sup: Region<'tcx>) {
        debug!("relate_regions(sub={:?}, sup={:?})", sub, sup);
        if is_free_or_static(sub) && is_free(sup) {
            self.relation.add(sub, sup)
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

/// The NLL region handling code represents free region relations in a
/// slightly different way; this trait allows functions to be abstract
/// over which version is in use.
pub trait FreeRegionRelations<'tcx> {
    /// Tests whether `r_a <= r_b`. Both must be free regions or
    /// `'static`.
    fn sub_free_regions(&self, shorter: ty::Region<'tcx>, longer: ty::Region<'tcx>) -> bool;
}

impl<'tcx> FreeRegionRelations<'tcx> for FreeRegionMap<'tcx> {
    fn sub_free_regions(&self,
                        r_a: Region<'tcx>,
                        r_b: Region<'tcx>)
                        -> bool {
        assert!(is_free_or_static(r_a) && is_free_or_static(r_b));
        if let ty::ReStatic = r_b {
            true // `'a <= 'static` is just always true, and not stored in the relation explicitly
        } else {
            r_a == r_b || self.relation.contains(&r_a, &r_b)
        }
    }
}

fn is_free(r: Region<'_>) -> bool {
    match *r {
        ty::ReEarlyBound(_) | ty::ReFree(_) => true,
        _ => false
    }
}

fn is_free_or_static(r: Region<'_>) -> bool {
    match *r {
        ty::ReStatic => true,
        _ => is_free(r),
    }
}

impl_stable_hash_for!(struct FreeRegionMap<'tcx> {
    relation
});

impl<'a, 'tcx> Lift<'tcx> for FreeRegionMap<'a> {
    type Lifted = FreeRegionMap<'tcx>;
    fn lift_to_tcx(&self, tcx: TyCtxt<'tcx>) -> Option<FreeRegionMap<'tcx>> {
        self.relation.maybe_map(|&fr| tcx.lift(&fr))
                     .map(|relation| FreeRegionMap { relation })
    }
}
