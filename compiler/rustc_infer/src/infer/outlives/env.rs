use rustc_data_structures::fx::FxIndexSet;
use rustc_data_structures::transitive_relation::TransitiveRelationBuilder;
use rustc_middle::{bug, ty};
use tracing::debug;

use super::explicit_outlives_bounds;
use crate::infer::GenericKind;
use crate::infer::free_regions::FreeRegionMap;
use crate::traits::query::OutlivesBound;

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
    /// FIXME: Your first reaction may be that this is a bit strange. `RegionBoundPairs`
    /// does not contain lifetimes, which are instead in the `FreeRegionMap`, and other
    /// known type outlives are stored in the `known_type_outlives` set. So why do we
    /// have these at all? It turns out that removing these and using `known_type_outlives`
    /// everywhere is just enough of a perf regression to matter. This can/should be
    /// optimized in the future, though.
    region_bound_pairs: RegionBoundPairs<'tcx>,
    known_type_outlives: Vec<ty::PolyTypeOutlivesPredicate<'tcx>>,
}

/// "Region-bound pairs" tracks outlives relations that are known to
/// be true, either because of explicit where-clauses like `T: 'a` or
/// because of implied bounds.
pub type RegionBoundPairs<'tcx> = FxIndexSet<ty::OutlivesPredicate<'tcx, GenericKind<'tcx>>>;

impl<'tcx> OutlivesEnvironment<'tcx> {
    /// Create a new `OutlivesEnvironment` from normalized outlives bounds.
    pub fn from_normalized_bounds(
        param_env: ty::ParamEnv<'tcx>,
        known_type_outlives: Vec<ty::PolyTypeOutlivesPredicate<'tcx>>,
        extra_bounds: impl IntoIterator<Item = OutlivesBound<'tcx>>,
    ) -> Self {
        let mut region_relation = TransitiveRelationBuilder::default();
        let mut region_bound_pairs = RegionBoundPairs::default();

        // Record relationships such as `T:'x` that don't go into the
        // free-region-map but which we use here.
        for outlives_bound in explicit_outlives_bounds(param_env).chain(extra_bounds) {
            debug!("add_outlives_bounds: outlives_bound={:?}", outlives_bound);
            match outlives_bound {
                OutlivesBound::RegionSubParam(r_a, param_b) => {
                    region_bound_pairs
                        .insert(ty::OutlivesPredicate(GenericKind::Param(param_b), r_a));
                }
                OutlivesBound::RegionSubAlias(r_a, alias_b) => {
                    region_bound_pairs
                        .insert(ty::OutlivesPredicate(GenericKind::Alias(alias_b), r_a));
                }
                OutlivesBound::RegionSubRegion(r_a, r_b) => match (r_a.kind(), r_b.kind()) {
                    (
                        ty::ReStatic | ty::ReEarlyParam(_) | ty::ReLateParam(_),
                        ty::ReStatic | ty::ReEarlyParam(_) | ty::ReLateParam(_),
                    ) => region_relation.add(r_a, r_b),
                    (ty::ReError(_), _) | (_, ty::ReError(_)) => {}
                    // FIXME(#109628): We shouldn't have existential variables in implied bounds.
                    // Panic here once the linked issue is resolved!
                    (ty::ReVar(_), _) | (_, ty::ReVar(_)) => {}
                    _ => bug!("add_outlives_bounds: unexpected regions: ({r_a:?}, {r_b:?})"),
                },
            }
        }

        OutlivesEnvironment {
            param_env,
            known_type_outlives,
            free_region_map: FreeRegionMap { relation: region_relation.freeze() },
            region_bound_pairs,
        }
    }

    pub fn free_region_map(&self) -> &FreeRegionMap<'tcx> {
        &self.free_region_map
    }

    pub fn region_bound_pairs(&self) -> &RegionBoundPairs<'tcx> {
        &self.region_bound_pairs
    }

    pub fn known_type_outlives(&self) -> &[ty::PolyTypeOutlivesPredicate<'tcx>] {
        &self.known_type_outlives
    }
}
