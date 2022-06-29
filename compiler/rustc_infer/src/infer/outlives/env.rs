use crate::infer::free_regions::FreeRegionMap;
use crate::infer::{GenericKind, InferCtxt};
use crate::traits::query::OutlivesBound;
use rustc_middle::ty::{self, ReEarlyBound, ReFree, ReVar, Region};

use super::explicit_outlives_bounds;

/// The `OutlivesEnvironment` collects information about what outlives
/// what in a given type-checking setting. For example, if we have a
/// where-clause like `where T: 'a` in scope, then the
/// `OutlivesEnvironment` would record that (in its
/// `region_bound_pairs` field). Similarly, it contains methods for
/// processing and adding implied bounds into the outlives
/// environment.
///
/// Other code at present does not typically take a
/// `&OutlivesEnvironment`, but rather takes some of its fields (e.g.,
/// `process_registered_region_obligations` wants the
/// region-bound-pairs). There is no mistaking it: the current setup
/// of tracking region information is quite scattered! The
/// `OutlivesEnvironment`, for example, needs to sometimes be combined
/// with the `middle::RegionRelations`, to yield a full picture of how
/// (lexical) lifetimes interact. However, I'm reluctant to do more
/// refactoring here, since the setup with NLL is quite different.
/// For example, NLL has no need of `RegionRelations`, and is solely
/// interested in the `OutlivesEnvironment`. -nmatsakis
#[derive(Clone)]
pub struct OutlivesEnvironment<'tcx> {
    pub param_env: ty::ParamEnv<'tcx>,
    free_region_map: FreeRegionMap<'tcx>,

    // Contains the implied region bounds in scope for our current body.
    //
    // Example:
    //
    // ```
    // fn foo<'a, 'b, T>(x: &'a T, y: &'b ()) {
    //   bar(x, y, |y: &'b T| { .. } // body B1)
    // } // body B0
    // ```
    //
    // Here, when checking the body B0, the list would be `[T: 'a]`, because we
    // infer that `T` must outlive `'a` from the implied bounds on the
    // fn declaration.
    //
    // For the body B1 however, the list would be `[T: 'a, T: 'b]`, because we
    // also can see that -- within the closure body! -- `T` must
    // outlive `'b`. This is not necessarily true outside the closure
    // body, since the closure may never be called.
    region_bound_pairs: RegionBoundPairs<'tcx>,
}

/// "Region-bound pairs" tracks outlives relations that are known to
/// be true, either because of explicit where-clauses like `T: 'a` or
/// because of implied bounds.
pub type RegionBoundPairs<'tcx> = Vec<(Region<'tcx>, GenericKind<'tcx>)>;

impl<'a, 'tcx> OutlivesEnvironment<'tcx> {
    pub fn new(param_env: ty::ParamEnv<'tcx>) -> Self {
        let mut env = OutlivesEnvironment {
            param_env,
            free_region_map: Default::default(),
            region_bound_pairs: Default::default(),
        };

        env.add_outlives_bounds(None, explicit_outlives_bounds(param_env));

        env
    }

    /// Borrows current value of the `free_region_map`.
    pub fn free_region_map(&self) -> &FreeRegionMap<'tcx> {
        &self.free_region_map
    }

    /// Borrows current `region_bound_pairs`.
    pub fn region_bound_pairs(&self) -> &RegionBoundPairs<'tcx> {
        &self.region_bound_pairs
    }

    /// Processes outlives bounds that are known to hold, whether from implied or other sources.
    ///
    /// The `infcx` parameter is optional; if the implied bounds may
    /// contain inference variables, it must be supplied, in which
    /// case we will register "givens" on the inference context. (See
    /// `RegionConstraintData`.)
    pub fn add_outlives_bounds<I>(
        &mut self,
        infcx: Option<&InferCtxt<'a, 'tcx>>,
        outlives_bounds: I,
    ) where
        I: IntoIterator<Item = OutlivesBound<'tcx>>,
    {
        // Record relationships such as `T:'x` that don't go into the
        // free-region-map but which we use here.
        for outlives_bound in outlives_bounds {
            debug!("add_outlives_bounds: outlives_bound={:?}", outlives_bound);
            match outlives_bound {
                OutlivesBound::RegionSubParam(r_a, param_b) => {
                    self.region_bound_pairs.push((r_a, GenericKind::Param(param_b)));
                }
                OutlivesBound::RegionSubProjection(r_a, projection_b) => {
                    self.region_bound_pairs.push((r_a, GenericKind::Projection(projection_b)));
                }
                OutlivesBound::RegionSubRegion(r_a, r_b) => {
                    if let (ReEarlyBound(_) | ReFree(_), ReVar(vid_b)) = (r_a.kind(), r_b.kind()) {
                        infcx
                            .expect("no infcx provided but region vars found")
                            .add_given(r_a, vid_b);
                    } else {
                        // In principle, we could record (and take
                        // advantage of) every relationship here, but
                        // we are also free not to -- it simply means
                        // strictly less that we can successfully type
                        // check. Right now we only look for things
                        // relationships between free regions. (It may
                        // also be that we should revise our inference
                        // system to be more general and to make use
                        // of *every* relationship that arises here,
                        // but presently we do not.)
                        self.free_region_map.relate_regions(r_a, r_b);
                    }
                }
            }
        }
    }
}
