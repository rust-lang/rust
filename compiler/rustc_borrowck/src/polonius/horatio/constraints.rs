use std::mem;
use std::ops::ControlFlow;

use rustc_index::IndexVec;
use rustc_index::bit_set::{DenseBitSet, SparseBitMatrix};
use rustc_middle::mir::{
    BasicBlock, Body, Location, Statement, StatementKind, Terminator, TerminatorKind,
};
use rustc_middle::ty::{TyCtxt, TypeVisitable};
use rustc_mir_dataflow::points::{DenseLocationMap, PointIndex};

use crate::constraints::OutlivesConstraint;
use crate::type_check::Locations;
use crate::{RegionInferenceContext, RegionVid};

/// The location-sensitive constraint graph.
///
/// This struct contains all outlives constraints. It internally distinguishes between global
/// constraints, which are in effect everywhere, and local constraints, which apply only at a
/// specific point. It can retrieve all constraints at a given point in constant time.
pub(crate) struct Constraints<'a, 'tcx> {
    /// A mapping from points to local outlives constraints (only active at a single point).
    ///
    /// At point `p`, we store all local outlives constraints that take effect at `p`. This means
    /// that their sup-region (`'a` in `'a: 'b`) is checked at `p`. As a consequence, time-travelling
    /// constraints that travel backwards in time are stored at the successor location(s) of the
    /// location from `constraint.locations`.
    local_constraints: IndexVec<PointIndex, Vec<LocalConstraint>>,

    /// A list of all outlives constraints that are active at every point in the CFG.
    global_constraints: Vec<GlobalConstraint>,

    tcx: TyCtxt<'tcx>,
    regioncx: &'a RegionInferenceContext<'tcx>,
    body: &'a Body<'tcx>,
    location_map: &'a DenseLocationMap,
}

/// A global outlives constraint that is active at every point in the CFG.
#[derive(Clone, Copy)]
struct GlobalConstraint {
    /// If we have the constraint `'a: 'b`, then `'a` is the sup and `'b` the sub.
    sup: RegionVid,
    /// If we have the constraint `'a: 'b`, then `'a` is the sup and `'b` the sub.
    sub: RegionVid,
}

/// A local outlives constraint that is only active at a single point in the CFG.
#[derive(Clone, Copy)]
struct LocalConstraint {
    /// If we have the constraint `'a: 'b`, then `'a` is the sup and `'b` the sub.
    sup: RegionVid,
    /// If we have the constraint `'a: 'b`, then `'a` is the sup and `'b` the sub.
    sub: RegionVid,

    /// If and how the constraint travels in time.
    time_travel: Option<(TimeTravelDirection, TimeTravelKind)>,
}

/// The direction of a time-travelling constraint.
///
/// Most local constraints apply at a single location in the CFG, but some flow either backwards or
/// forwards in time—to the previous or next location. For instance, given the constraint `'a: 'b`
/// at location `l`, if the constraint flows forwards, then `'b` is active at the successor of `l`
/// if `'a` is active at `l`. Conversely, if it flows backwards, `'b` is active at the predecessor
/// of `l` if `'a` is active at `l`.
#[derive(Debug, Copy, Clone)]
enum TimeTravelDirection {
    /// The constraint flows backwards in time.
    ///
    /// `'a: 'b` at location `l` means that `'b` is active at the predecessor of `l` if `'a` is
    /// active at `l`.
    Backwards,

    /// The constraint flows forwards in time.
    ///
    /// `'a: 'b` at location `l` means that `'b` is active at the successor of `l` if `'a` is
    /// active at `l`.
    Forwards,
}

/// Whether a time-travelling constraint stays within the same block or crosses block boundaries.
///
/// The constraint's "location" (or source location) is the point in the CFG where the sup-region is
/// checked. For `'a: 'b`, this is where `'a` is checked. The "target location" is where `'b` becomes
/// active. The source and target locations are either the same, or—if the constraint is
/// time-travelling—adjacent.
#[derive(Debug, Copy, Clone)]
enum TimeTravelKind {
    /// The constraint travels within the same block.
    ///
    /// Suppose we have the constraint `'a: 'b`. If it travels backwards, then `'a` cannot be checked
    /// at the first location in a block, or `'b` would be active in a preceding block. Similarly,
    /// if it travels forwards, `'a` cannot be checked at the terminator.
    IntraBlock,

    /// The constraint travels across block boundaries.
    ///
    /// The source and target locations are in different blocks. Since they must be adjacent, a
    /// forward-travelling constraint implies the source location is a terminator and the target is
    /// the first location of a block. Conversely, a backward-travelling constraint implies the
    /// source is the first location of a block and the target is a terminator.
    InterBlock {
        /// The block containing the target location.
        ///
        /// The statement index of the target location is `0` if the constraint travels forwards,
        /// or the index of the terminator if it travels backwards.
        target_block: BasicBlock,
    },
}

#[derive(Default)]
pub(crate) struct TimeTravellingRegions {
    /// Regions travelling to the proceeding statement within the same block.
    pub to_prev_stmt: Option<DenseBitSet<RegionVid>>,

    /// Regions travelling to proceeding blocks. Only applicable at the first statement of a block.
    pub to_predecessor_blocks: Option<SparseBitMatrix<BasicBlock, RegionVid>>,

    /// Regions travelling to the next statement within the same block. Not applicable for
    /// terminators.
    pub to_next_loc: Option<DenseBitSet<RegionVid>>,

    /// Regions travelling to succeeding blocks. Only applicable at the first statement of a block.
    pub to_successor_blocks: Option<SparseBitMatrix<BasicBlock, RegionVid>>,
}

impl TimeTravellingRegions {
    fn add_within_block(
        &mut self,
        regioncx: &RegionInferenceContext<'_>,
        region: RegionVid,
        direction: TimeTravelDirection,
    ) {
        match direction {
            TimeTravelDirection::Forwards => {
                self.to_next_loc
                    .get_or_insert_with(|| DenseBitSet::new_empty(regioncx.num_regions()))
                    .insert(region);
            }
            TimeTravelDirection::Backwards => {
                self.to_prev_stmt
                    .get_or_insert_with(|| DenseBitSet::new_empty(regioncx.num_regions()))
                    .insert(region);
            }
        }
    }

    fn add_to_predecessor_block(
        &mut self,
        regioncx: &RegionInferenceContext<'_>,
        region: RegionVid,
        preceeding_block: BasicBlock,
    ) {
        self.to_predecessor_blocks
            .get_or_insert_with(|| SparseBitMatrix::new(regioncx.num_regions()))
            .insert(preceeding_block, region);
    }

    fn add_to_successor_block(
        &mut self,
        regioncx: &RegionInferenceContext<'_>,
        region: RegionVid,
        succeeding_block: BasicBlock,
    ) {
        self.to_successor_blocks
            .get_or_insert_with(|| SparseBitMatrix::new(regioncx.num_regions()))
            .insert(succeeding_block, region);
    }
}

impl<'a, 'tcx> Constraints<'a, 'tcx> {
    pub(crate) fn new(
        tcx: TyCtxt<'tcx>,
        regioncx: &'a RegionInferenceContext<'tcx>,
        body: &'a Body<'tcx>,
        location_map: &'a DenseLocationMap,
    ) -> Self {
        Self {
            local_constraints: IndexVec::from_elem_n(vec![], location_map.num_points()),
            global_constraints: vec![],
            tcx,
            regioncx,
            body,
            location_map,
        }
    }

    pub(crate) fn add_constraint(&mut self, constraint: &OutlivesConstraint<'tcx>) {
        match constraint.locations {
            Locations::Single(location) => {
                let (source_location, time_travel) = if let Some(stmt) =
                    self.body[location.block].statements.get(location.statement_index)
                {
                    match self.time_travel_at_statement(&constraint, stmt) {
                        Some(t @ TimeTravelDirection::Forwards) => {
                            (location, Some((t, TimeTravelKind::IntraBlock)))
                        }
                        Some(t @ TimeTravelDirection::Backwards) => (
                            location.successor_within_block(),
                            Some((t, TimeTravelKind::IntraBlock)),
                        ),
                        None => (location, None),
                    }
                } else {
                    debug_assert_eq!(
                        location.statement_index,
                        self.body[location.block].statements.len()
                    );
                    let terminator = self.body[location.block].terminator();
                    match self.time_travel_at_terminator(&constraint, terminator) {
                        Some((t @ TimeTravelDirection::Forwards, target_block)) => {
                            (location, Some((t, TimeTravelKind::InterBlock { target_block })))
                        }
                        Some((t @ TimeTravelDirection::Backwards, source_block)) => (
                            Location { block: source_block, statement_index: 0 },
                            Some((t, TimeTravelKind::InterBlock { target_block: location.block })),
                        ),
                        None => (location, None),
                    }
                };

                let point = self.location_map.point_from_location(source_location);
                self.local_constraints[point].push(LocalConstraint {
                    sup: constraint.sup,
                    sub: constraint.sub,
                    time_travel,
                });
            }
            Locations::All(_) => {
                self.global_constraints
                    .push(GlobalConstraint { sup: constraint.sup, sub: constraint.sub });
            }
        }
    }

    /// Checks if and in which direction a constraint at a statement travels in time.
    fn time_travel_at_statement(
        &self,
        constraint: &OutlivesConstraint<'tcx>,
        statement: &Statement<'tcx>,
    ) -> Option<TimeTravelDirection> {
        match &statement.kind {
            StatementKind::Assign(box (lhs, _)) => {
                let lhs_ty = self.body.local_decls[lhs.local].ty;
                self.compute_constraint_direction(constraint, &lhs_ty)
            }
            _ => None,
        }
    }

    /// Check if/how an outlives constraint travels in time at a terminator.
    ///
    /// Returns an `Option` of the pair `(direction, block)`. Where `direction` is a
    /// `TimeTravelDirection` and `block` is the target or source block of a forwards or backwards
    /// travelling constraint respectively.
    fn time_travel_at_terminator(
        &self,
        constraint: &OutlivesConstraint<'tcx>,
        terminator: &Terminator<'tcx>,
    ) -> Option<(TimeTravelDirection, BasicBlock)> {
        // FIXME: check if other terminators need the same handling as `Call`s, in particular
        // Assert/Yield/Drop. A handful of tests are failing with Drop related issues, as well as some
        // coroutine tests, and that may be why.
        match &terminator.kind {
            // FIXME: also handle diverging calls.
            TerminatorKind::Call { destination, target: Some(target_block), .. } => {
                // Calls are similar to assignments, and thus follow the same pattern. If there is a
                // target for the call we also relate what flows into the destination here to entry to
                // that successor.
                let destination_ty = destination.ty(&self.body.local_decls, self.tcx);
                self.compute_constraint_direction(constraint, &destination_ty)
                    .map(|t| (t, *target_block))
            }
            _ => None,
        }
    }

    /// For a given outlives constraint and CFG edge, returns the localized constraint with the
    /// appropriate `from`-`to` direction. This is computed according to whether the constraint flows to
    /// or from a free region in the given `value`, some kind of result for an effectful operation, like
    /// the LHS of an assignment.
    fn compute_constraint_direction(
        &self,
        constraint: &OutlivesConstraint<'tcx>,
        value: &impl TypeVisitable<TyCtxt<'tcx>>,
    ) -> Option<TimeTravelDirection> {
        // FIXME: There seem to be cases where both sub and sup appear in the free regions.

        self.tcx.for_each_free_region_until(value, |region| {
            let region = self.regioncx.universal_regions().to_region_vid(region);
            if region == constraint.sub {
                // This constraint flows into the result, its effects start becoming visible on exit.
                ControlFlow::Break(TimeTravelDirection::Forwards)
            } else if region == constraint.sup {
                // This constraint flows from the result, its effects start becoming visible on exit.
                ControlFlow::Break(TimeTravelDirection::Backwards)
            } else {
                ControlFlow::Continue(())
            }
        })
    }

    /// Given a set of regions at a certain point in the CFG, add all regions induced by outlives
    /// constraints at that point  to the set. Additionally, all regions arising from time
    /// travelling constraints will be collected and returned.
    ///
    /// If we have the set `{'a, 'b}`, and we have the following constraints:
    /// - `'a: 'c`
    /// - `'b: 'd`
    /// - `'d: 'e`
    /// Then `'c`, `'d` and `'e` will be added to the set.
    ///
    /// Also, any time travelling constraints implied by any of these five regions would be
    /// collected and returned in the `TimeTravellingRegions` struct.
    pub(crate) fn add_dependent_regions_at_point(
        &self,
        point: PointIndex,
        regions: &mut DenseBitSet<RegionVid>,
    ) -> TimeTravellingRegions {
        // This function will loop until there are no more regions to add. It will keep a set of
        // regions that has not been considered yet (the `to_check` variable). At each iteration of
        // the main loop, It'll walk through all constraints at this point and all global
        // constraints. Any regions implied from the `to_check` set  will be put in the
        // `to_check_next_round` set. When all constraints has been considered, the `to_check` set
        // will be cleared. It will be swaped with the `to_check_next_round` set, and then the main
        // loop runs again. It'll stop when there are no more regions to check.
        //
        // The time travelling constraints will be treated differently. Regions implied by time
        // travelling constraints will be collected in an instance of the `TimeTravellingRegions`
        // struct.

        let mut to_check = regions.clone();
        let mut to_check_next_round = DenseBitSet::new_empty(self.regioncx.num_regions());
        let mut time_travelling_regions = TimeTravellingRegions::default();

        // Loop till the fixpoint: when there are no more regions to add.
        while !to_check.is_empty() {
            // Loop through all global constraints.
            for constraint in &self.global_constraints {
                if !to_check.contains(constraint.sup) {
                    continue;
                }
                if regions.insert(constraint.sub) {
                    to_check_next_round.insert(constraint.sub);
                }
            }

            // Loop through all local constraints.
            for constraint in &self.local_constraints[point] {
                if !to_check.contains(constraint.sup) {
                    continue;
                }

                // Check if the constraint is travelling in time.
                if let Some((travel_direction, travel_kind)) = constraint.time_travel {
                    match (travel_direction, travel_kind) {
                        (direction, TimeTravelKind::IntraBlock) => time_travelling_regions
                            .add_within_block(self.regioncx, constraint.sub, direction),
                        (
                            TimeTravelDirection::Forwards,
                            TimeTravelKind::InterBlock { target_block },
                        ) => time_travelling_regions.add_to_successor_block(
                            self.regioncx,
                            constraint.sub,
                            target_block,
                        ),
                        (
                            TimeTravelDirection::Backwards,
                            TimeTravelKind::InterBlock { target_block },
                        ) => time_travelling_regions.add_to_predecessor_block(
                            self.regioncx,
                            constraint.sub,
                            target_block,
                        ),
                    }

                    // If the region is time travelling we should not add it to
                    // `regions`.
                    continue;
                }

                if regions.insert(constraint.sub) {
                    to_check_next_round.insert(constraint.sub);
                }
            }

            mem::swap(&mut to_check, &mut to_check_next_round);
            to_check_next_round.clear();
        }

        time_travelling_regions
    }

    /// Given a set of regions, add all regions induced by outlives constraints at any point in the
    /// CFG to the set.
    ///
    /// If we have the set `{'a, 'b}`, and we have the following constraints:
    /// - `'a: 'c`
    /// - `'b: 'd`
    /// - `'d: 'e`
    /// Then `'c`, `'d` and `'e` will be added to the set.
    #[inline(never)] // FIXME: Remove this.
    pub(crate) fn add_dependent_regions(&self, regions: &mut DenseBitSet<RegionVid>) {
        // This function will loop until there are no more regions to add. It will keep a set of
        // regions that has not been considered yet (the `to_check` variable). At each iteration of
        // the main loop, It'll walk through all constraints, both global and local. Any regions
        // implied from the `to_check` set  will be put in the `to_check_next_round` set. When all
        // constraints has been considered, the `to_check` set will be cleared. It will be swaped
        // with the `to_check_next_round` set, and then the main loop runs again. It'll stop when
        // there are no more regions to check.
        //
        // The time travelling constraints will not be treated differently in this function.

        let mut to_check = regions.clone();
        let mut to_check_next_round = DenseBitSet::new_empty(self.regioncx.num_regions());

        // Loop till the fixpoint: when there are no more regions to add.
        while !to_check.is_empty() {
            // Loop through all global constraints.
            for constraint in &self.global_constraints {
                if !to_check.contains(constraint.sup) {
                    continue;
                }
                if regions.insert(constraint.sub) {
                    to_check_next_round.insert(constraint.sub);
                }
            }

            // Loop through all local constraints.
            for constraint in self.local_constraints.iter().flatten() {
                if !to_check.contains(constraint.sup) {
                    continue;
                }
                if regions.insert(constraint.sub) {
                    to_check_next_round.insert(constraint.sub);
                }
            }

            mem::swap(&mut to_check, &mut to_check_next_round);
            to_check_next_round.clear();
        }
    }

    /// Like `add_dependent_regions()` but with constraints reversed.
    // FIXME: Could these functions be merged to avoid code duplication.
    #[inline(never)] // FIXME: Remove this.
    pub(crate) fn add_dependent_regions_reversed(&self, regions: &mut DenseBitSet<RegionVid>) {
        // See the `add_dependent_regions()` function for an explonation of the code. The functions
        // are identical except that we swapped sub and sup.

        let mut to_check = regions.clone();
        let mut to_check_next_round = DenseBitSet::new_empty(self.regioncx.num_regions());

        // Loop till the fixpoint: when there are no more regions to add.
        while !to_check.is_empty() {
            // Loop through all global constraints.
            for constraint in &self.global_constraints {
                if !to_check.contains(constraint.sub) {
                    continue;
                }
                if regions.insert(constraint.sup) {
                    to_check_next_round.insert(constraint.sup);
                }
            }

            // Loop through all local constraints.
            for constraint in self.local_constraints.iter().flatten() {
                if !to_check.contains(constraint.sub) {
                    continue;
                }
                if regions.insert(constraint.sup) {
                    to_check_next_round.insert(constraint.sup);
                }
            }

            mem::swap(&mut to_check, &mut to_check_next_round);
            to_check_next_round.clear();
        }
    }
}
