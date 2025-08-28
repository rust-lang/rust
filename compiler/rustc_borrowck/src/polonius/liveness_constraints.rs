use std::collections::BTreeMap;

use rustc_index::bit_set::SparseBitMatrix;
use rustc_middle::mir::{Body, Location};
use rustc_middle::ty::relate::{self, Relate, RelateResult, TypeRelation};
use rustc_middle::ty::{self, RegionVid, Ty, TyCtxt, TypeVisitable};
use rustc_mir_dataflow::points::PointIndex;

use super::{
    ConstraintDirection, LocalizedOutlivesConstraint, LocalizedOutlivesConstraintSet,
    PoloniusLivenessContext,
};
use crate::region_infer::values::LivenessValues;
use crate::universal_regions::UniversalRegions;

impl PoloniusLivenessContext {
    /// Record the variance of each region contained within the given value.
    pub(crate) fn record_live_region_variance<'tcx>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        universal_regions: &UniversalRegions<'tcx>,
        value: impl TypeVisitable<TyCtxt<'tcx>> + Relate<TyCtxt<'tcx>>,
    ) {
        let mut extractor = VarianceExtractor {
            tcx,
            ambient_variance: ty::Variance::Covariant,
            directions: &mut self.live_region_variances,
            universal_regions,
        };
        extractor.relate(value, value).expect("Can't have a type error relating to itself");
    }
}

/// Propagate loans throughout the CFG: for each statement in the MIR, create localized outlives
/// constraints for loans that are propagated to the next statements.
pub(super) fn create_liveness_constraints<'tcx>(
    body: &Body<'tcx>,
    liveness: &LivenessValues,
    live_regions: &SparseBitMatrix<PointIndex, RegionVid>,
    live_region_variances: &BTreeMap<RegionVid, ConstraintDirection>,
    universal_regions: &UniversalRegions<'tcx>,
    localized_outlives_constraints: &mut LocalizedOutlivesConstraintSet,
) {
    for (block, bb) in body.basic_blocks.iter_enumerated() {
        let statement_count = bb.statements.len();
        for statement_index in 0..=statement_count {
            let current_location = Location { block, statement_index };
            let current_point = liveness.point_from_location(current_location);

            if statement_index < statement_count {
                // Intra-block edges, straight line constraints from each point to its successor
                // within the same block.
                let next_location = Location { block, statement_index: statement_index + 1 };
                let next_point = liveness.point_from_location(next_location);
                propagate_loans_between_points(
                    current_point,
                    next_point,
                    live_regions,
                    live_region_variances,
                    universal_regions,
                    localized_outlives_constraints,
                );
            } else {
                // Inter-block edges, from the block's terminator to each successor block's entry
                // point.
                for successor_block in bb.terminator().successors() {
                    let next_location = Location { block: successor_block, statement_index: 0 };
                    let next_point = liveness.point_from_location(next_location);
                    propagate_loans_between_points(
                        current_point,
                        next_point,
                        live_regions,
                        live_region_variances,
                        universal_regions,
                        localized_outlives_constraints,
                    );
                }
            }
        }
    }
}

/// Propagate loans within a region between two points in the CFG, if that region is live at both
/// the source and target points.
fn propagate_loans_between_points(
    current_point: PointIndex,
    next_point: PointIndex,
    live_regions: &SparseBitMatrix<PointIndex, RegionVid>,
    live_region_variances: &BTreeMap<RegionVid, ConstraintDirection>,
    universal_regions: &UniversalRegions<'_>,
    localized_outlives_constraints: &mut LocalizedOutlivesConstraintSet,
) {
    // Universal regions are semantically live at all points.
    // Note: we always have universal regions but they're not always (or often) involved in the
    // subset graph. For now, we emit all their edges unconditionally, but some of these subgraphs
    // will be disconnected from the rest of the graph and thus, unnecessary.
    //
    // FIXME: only emit the edges of universal regions that existential regions can reach.
    for region in universal_regions.universal_regions_iter() {
        localized_outlives_constraints.push(LocalizedOutlivesConstraint {
            source: region,
            from: current_point,
            target: region,
            to: next_point,
        });
    }

    let Some(next_live_regions) = live_regions.row(next_point) else {
        // There are no constraints to add: there are no live regions at the next point.
        return;
    };

    for region in next_live_regions.iter() {
        // `region` could be live at the current point, and is live at the next point: add a
        // constraint between them, according to variance.
        if let Some(&direction) = live_region_variances.get(&region) {
            add_liveness_constraint(
                region,
                current_point,
                next_point,
                direction,
                localized_outlives_constraints,
            );
        } else {
            // Note: there currently are cases related to promoted and const generics, where we
            // don't yet have variance information (possibly about temporary regions created when
            // typeck sanitizes the promoteds). Until that is done, we conservatively fallback to
            // maximizing reachability by adding a bidirectional edge here. This will not limit
            // traversal whatsoever, and thus propagate liveness when needed.
            //
            // FIXME: add the missing variance information and remove this fallback bidirectional
            // edge.
            let fallback = ConstraintDirection::Bidirectional;
            add_liveness_constraint(
                region,
                current_point,
                next_point,
                fallback,
                localized_outlives_constraints,
            );
        }
    }
}

/// Adds `LocalizedOutlivesConstraint`s between two connected points, according to the given edge
/// direction.
fn add_liveness_constraint(
    region: RegionVid,
    current_point: PointIndex,
    next_point: PointIndex,
    direction: ConstraintDirection,
    localized_outlives_constraints: &mut LocalizedOutlivesConstraintSet,
) {
    match direction {
        ConstraintDirection::Forward => {
            // Covariant cases: loans flow in the regular direction, from the current point to the
            // next point.
            localized_outlives_constraints.push(LocalizedOutlivesConstraint {
                source: region,
                from: current_point,
                target: region,
                to: next_point,
            });
        }
        ConstraintDirection::Backward => {
            // Contravariant cases: loans flow in the inverse direction, from the next point to the
            // current point.
            localized_outlives_constraints.push(LocalizedOutlivesConstraint {
                source: region,
                from: next_point,
                target: region,
                to: current_point,
            });
        }
        ConstraintDirection::Bidirectional => {
            // For invariant cases, loans can flow in both directions: we add both edges.
            localized_outlives_constraints.push(LocalizedOutlivesConstraint {
                source: region,
                from: current_point,
                target: region,
                to: next_point,
            });
            localized_outlives_constraints.push(LocalizedOutlivesConstraint {
                source: region,
                from: next_point,
                target: region,
                to: current_point,
            });
        }
    }
}

/// Extracts variances for regions contained within types. Follows the same structure as
/// `rustc_infer`'s `Generalizer`: we try to relate a type with itself to track and extract the
/// variances of regions.
struct VarianceExtractor<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    ambient_variance: ty::Variance,
    directions: &'a mut BTreeMap<RegionVid, ConstraintDirection>,
    universal_regions: &'a UniversalRegions<'tcx>,
}

impl<'tcx> VarianceExtractor<'_, 'tcx> {
    fn record_variance(&mut self, region: ty::Region<'tcx>, variance: ty::Variance) {
        // We're only interested in the variance of vars and free regions.
        //
        // Note: even if we currently bail for two cases of unexpected region kinds here, missing
        // variance data is not a soundness problem: the regions with missing variance will still be
        // present in the constraint graph as they are live, and liveness edges construction has a
        // fallback for this case.
        //
        // FIXME: that being said, we need to investigate these cases better to not ignore regions
        // in general.
        if region.is_bound() {
            // We ignore these because they cannot be turned into the vids we need.
            return;
        }

        if region.is_erased() {
            // These cannot be turned into a vid either, and we also ignore them: the fact that they
            // show up here looks like either an issue upstream or a combination with unexpectedly
            // continuing compilation too far when we're in a tainted by errors situation.
            //
            // FIXME: investigate the `generic_const_exprs` test that triggers this issue,
            // `ui/const-generics/generic_const_exprs/issue-97047-ice-2.rs`
            return;
        }

        let direction = match variance {
            ty::Covariant => ConstraintDirection::Forward,
            ty::Contravariant => ConstraintDirection::Backward,
            ty::Invariant => ConstraintDirection::Bidirectional,
            ty::Bivariant => {
                // We don't add edges for bivariant cases.
                return;
            }
        };

        let region = self.universal_regions.to_region_vid(region);
        self.directions
            .entry(region)
            .and_modify(|entry| {
                // If there's already a recorded direction for this region, we combine the two:
                // - combining the same direction is idempotent
                // - combining different directions is trivially bidirectional
                if entry != &direction {
                    *entry = ConstraintDirection::Bidirectional;
                }
            })
            .or_insert(direction);
    }
}

impl<'tcx> TypeRelation<TyCtxt<'tcx>> for VarianceExtractor<'_, 'tcx> {
    fn cx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn relate_with_variance<T: Relate<TyCtxt<'tcx>>>(
        &mut self,
        variance: ty::Variance,
        _info: ty::VarianceDiagInfo<TyCtxt<'tcx>>,
        a: T,
        b: T,
    ) -> RelateResult<'tcx, T> {
        let old_ambient_variance = self.ambient_variance;
        self.ambient_variance = self.ambient_variance.xform(variance);
        let r = self.relate(a, b)?;
        self.ambient_variance = old_ambient_variance;
        Ok(r)
    }

    fn tys(&mut self, a: Ty<'tcx>, b: Ty<'tcx>) -> RelateResult<'tcx, Ty<'tcx>> {
        assert_eq!(a, b); // we are misusing TypeRelation here; both LHS and RHS ought to be ==
        relate::structurally_relate_tys(self, a, b)
    }

    fn regions(
        &mut self,
        a: ty::Region<'tcx>,
        b: ty::Region<'tcx>,
    ) -> RelateResult<'tcx, ty::Region<'tcx>> {
        assert_eq!(a, b); // we are misusing TypeRelation here; both LHS and RHS ought to be ==
        self.record_variance(a, self.ambient_variance);
        Ok(a)
    }

    fn consts(
        &mut self,
        a: ty::Const<'tcx>,
        b: ty::Const<'tcx>,
    ) -> RelateResult<'tcx, ty::Const<'tcx>> {
        assert_eq!(a, b); // we are misusing TypeRelation here; both LHS and RHS ought to be ==
        relate::structurally_relate_consts(self, a, b)
    }

    fn binders<T>(
        &mut self,
        a: ty::Binder<'tcx, T>,
        _: ty::Binder<'tcx, T>,
    ) -> RelateResult<'tcx, ty::Binder<'tcx, T>>
    where
        T: Relate<TyCtxt<'tcx>>,
    {
        self.relate(a.skip_binder(), a.skip_binder())?;
        Ok(a)
    }
}
