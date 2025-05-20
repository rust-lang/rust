use rustc_data_structures::fx::FxHashMap;
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::{BasicBlockData, Location};

use super::constraints::TimeTravellingRegions;
use super::{
    BorrowContext, KillsCache, PoloniusBlock, PoloniusContext, is_killed, my_println,
    remove_dead_regions,
};
use crate::RegionVid;

/// The main struct for the location-sensitive analysis.
///
/// Whenever the location-insensitive analysis (NLL) fails to prove that a loan is not active at a
/// given point, and we are about to emit an error, this location-sensitive analysis is triggered to
/// compute a definitive answer to the question: "Is this loan truly active at this location?"
///
/// The computation traverses the relevant parts of the CFG until a conclusive answer can be
/// determined, then pauses. The same instance of this struct can be reused to check whether the
/// loan is active at another point, for the same loan.
pub(super) struct LocationSensitiveAnalysis {
    /// All the nodes in the location sensitive graph. This graph is a subgraph of the CFG.
    pub nodes: FxHashMap<Location, Node>,

    /// Whether the computation is finished.
    ///
    /// If this is `true`, the loan's activeness has already been computed for all relevant
    /// locations, and it is sufficient to query the [Self::nodes] map to determine whether the loan
    /// is active at a specific location.
    pub is_finished: bool,

    /// A stack of nodes that should be checked next.
    primary_stack: Vec<Location>,

    /// A secondary stack. This will only be popped when the primary stack is empty.
    secondary_stack: Vec<Location>,
}

/// A node in the location-sensitive analysis.
pub(super) struct Node {
    /// The set of regions associated with the loan at this location.
    ///
    /// This set may grow on subsequent visits to this node, but it will never shrink. If it is
    /// empty, the traversal should not proceed to this node's neighbours.
    associated_regions: DenseBitSet<RegionVid>,

    /// The regions that were added to [Self::associated_regions] last time this node was added to
    /// the stack.
    ///
    /// This is only for optimization purposes, we don't want to check the regions that have already
    /// been checked before.
    added_regions: Option<DenseBitSet<RegionVid>>,

    /// Whether this location is reachable by forward edges from the loan's introduction point in
    /// the localized constraint graph.
    // FIXME: This comment seems strange.
    reachable_by_loan: bool,

    /// Whether the loan is active at this point.
    pub is_active: bool,

    /// Whether this node has been added to the stack for processing.
    added_to_stack: bool,
}

impl LocationSensitiveAnalysis {
    pub(super) fn new(bcx: BorrowContext<'_, '_, '_>) -> Self {
        // Put the loan's initial region in a set.
        let mut initial_region_set = DenseBitSet::new_empty(bcx.pcx.regioncx.num_regions());
        initial_region_set.insert(bcx.borrow.region);

        let mut nodes = FxHashMap::default();
        // Add the node at the loan's reserve location.
        nodes.insert(
            bcx.borrow.reserve_location,
            Node {
                associated_regions: DenseBitSet::new_empty(bcx.pcx.regioncx.num_regions()),
                added_regions: Some(initial_region_set),
                reachable_by_loan: false,
                is_active: false,
                added_to_stack: true,
            },
        );

        Self {
            primary_stack: vec![bcx.borrow.reserve_location],
            secondary_stack: vec![],
            nodes,
            is_finished: false,
        }
    }

    /// Compute the necessary nodes to conclude if a loan is active at `target_location`.
    #[inline(never)] // FIXME: Remove this.
    pub(super) fn compute(
        &mut self,
        bcx: BorrowContext<'_, '_, '_>,
        kills_cache: &mut KillsCache,
        target_location: Location,
        live_paths: DenseBitSet<PoloniusBlock>,
    ) -> bool {
        my_println!("Checking {:?} at {:?}", bcx.borrow_idx, target_location);
        debug_assert!(
            !self.is_finished,
            "If the location sensitive analysis is finished you should just query `LocationInsensitiveAnalysis::nodes` instead."
        );

        // Pop a node from the stack until it is empty.
        while let Some(location) = self.primary_stack.pop().or_else(|| self.secondary_stack.pop()) {
            let point = bcx.pcx.location_map.point_from_location(location);
            let block_data = &bcx.pcx.body[location.block];

            // Debugging: Print the current location and statement/expression.
            if let Some(stmt) = block_data.statements.get(location.statement_index) {
                my_println!("  {:?}: {:?}", location, stmt);
            } else {
                my_println!("  {:?}: {:?}", location, block_data.terminator().kind);
            }

            // Fetch the current node.
            let Node {
                associated_regions,
                added_regions,
                reachable_by_loan,
                is_active,
                added_to_stack,
            } = self.nodes.get_mut(&location).unwrap();
            let reachable_by_loan = *reachable_by_loan; // Make copy.

            debug_assert!(*added_to_stack);
            *added_to_stack = false;

            let time_travelling_regions = if let Some(mut added_regions) = added_regions.take() {
                debug_assert!(
                    !added_regions.is_empty(),
                    "added_regions should never be empty, in that case it should be `None`."
                );
                debug_assert!(
                    added_regions.iter().all(|r| !associated_regions.contains(r)),
                    "`added_regions` and `associated_regions` should be disjunct."
                );

                // Traverse the location-sensitive constraint graph at this point, adding any
                // regions reachable from the ones in `added_regions`. All time-travelling regions
                // encountered will be returned and stored in this variable.
                let time_travelling_regions = bcx
                    .pcx
                    .cache()
                    .constraints
                    .add_dependent_regions_at_point(point, &mut added_regions);

                // FIXME: Just debugging of the time-travelling regions.
                if let Some(tf) = &time_travelling_regions.to_next_loc {
                    my_println!("    Forward time travellers: {:?}", tf);
                }
                if let Some(tf) = &time_travelling_regions.to_prev_stmt {
                    my_println!("    Backward time travellers: {:?}", tf);
                }
                if let Some(x) = &time_travelling_regions.to_predecessor_blocks {
                    my_println!("    To preceeding blocks: {:?}", x);
                }
                if let Some(x) = &time_travelling_regions.to_successor_blocks {
                    my_println!("    To succeeding blocks: {:?}", x);
                }

                // Incorporate the added regions into `associated_regions`.
                associated_regions.union(&added_regions);
                my_println!("    Regions: {:?}", associated_regions);

                Some(time_travelling_regions)
            } else {
                my_println!("Nothing new here.");
                // FIXME: This should be unnecessary if we don't track kills.
                if reachable_by_loan {
                    // FIXME: This is just a hack.
                    let mut associated_regions = associated_regions.clone();
                    remove_dead_regions(bcx.pcx, location, &mut associated_regions);
                    if associated_regions.is_empty() {
                        my_println!("  Loan killed.");
                        continue;
                    }
                } else {
                    continue;
                }

                None
            };

            // Remove the dead regions from `associated_regions`.
            let mut associated_regions = associated_regions.clone();
            remove_dead_regions(bcx.pcx, location, &mut associated_regions);

            // If `associated_regions` is not empty and the node is reachable from the loan's
            // introduction point in the location-sensitive graph, then the loan is active.
            if reachable_by_loan && !associated_regions.is_empty() {
                *is_active = true;
                my_println!("    In scope at {location:?}");
            }

            // Check if the loan is killed.
            let is_killed = is_killed(bcx, kills_cache, location);

            // If the loan is killed at this location, and this location is reachable from the
            // loan's reserve location, then we should not add any of this node’s neighbours to the
            // stack.
            if is_killed && bcx.pcx.is_predecessor(bcx.borrow.reserve_location, location) {
                continue;
            }

            let successor_reachable_by_loan =
                !is_killed && reachable_by_loan || location == bcx.borrow.reserve_location;

            // Necessary to make the borrow checker happy.
            let is_active = *is_active;

            // Visit all the neighbours of this node—that is, both predecessors and successors—and
            // potentially add more associated regions to them.
            visit_adjacent_locations(
                bcx.pcx,
                block_data,
                location,
                time_travelling_regions,
                |new_location, time_travellers, is_forward| {
                    // Get or add a new node at this location.
                    let new_node = self.nodes.entry(new_location).or_insert_with(|| Node {
                        associated_regions: DenseBitSet::new_empty(bcx.pcx.regioncx.num_regions()),
                        added_regions: None,
                        reachable_by_loan: false,
                        is_active: false,
                        added_to_stack: false,
                    });

                    // Keep track of whether `new_node` has changed.
                    let mut new_node_changed = false;

                    // If we are going forwards, we need to propagate reachability for the loan.
                    if is_forward && successor_reachable_by_loan && !new_node.reachable_by_loan {
                        new_node.reachable_by_loan = true;
                        // `reachable_by_loan` was `false` before on `new_node` but has now been
                        // changed to `true`.
                        new_node_changed = true;
                    }

                    // Check if any regions should be added to `new_node`.
                    let mut added_regions = associated_regions.clone();
                    if !is_forward {
                        // FIXME: Now we only ignore the universal regions when going backwards. Actually we might need to record the variance of all regions, but the tests seem to pass in this way.
                        added_regions.subtract(&bcx.pcx.cache().universal_regions);
                    }

                    remove_dead_regions(bcx.pcx, new_location, &mut added_regions);

                    if let Some(time_travellers) = time_travellers {
                        added_regions.union(time_travellers);
                    }

                    // Subtract the already associated regions from `added_regions` so they become
                    // disjunct.
                    added_regions.subtract(&new_node.associated_regions);

                    if !added_regions.is_empty() {
                        if let Some(already_added_regions) = new_node.added_regions.as_mut() {
                            already_added_regions.union(&added_regions);
                        } else {
                            new_node.added_regions = Some(added_regions);
                        }
                        new_node_changed = true;
                    }

                    if new_node_changed && !new_node.added_to_stack {
                        if !is_forward
                            || live_paths.contains(PoloniusBlock::from_location(bcx, new_location))
                        {
                            self.primary_stack.push(new_location);
                        } else {
                            self.secondary_stack.push(new_location);
                        }
                        new_node.added_to_stack = true;
                    }
                },
            );

            if is_active && location == target_location {
                return true;
            }
        }

        // The stack is empty so the location sensitive analysis is complete.
        self.is_finished = true;

        // Fetch the result.
        self.nodes.get(&target_location).is_some_and(|x| x.is_active)
    }
}

/// This is a very specific function used in [`LocationInsensitiveAnalysis::compute()`] to visit all
///
/// predecessors and successors of a node. One could argue that it shouldn’t be a separate function
/// and should just be hardcoded, but that led to a ton of repetitive code.
#[inline]
fn visit_adjacent_locations(
    pcx: &PoloniusContext<'_, '_>,
    block_data: &BasicBlockData<'_>,
    location: Location,
    maybe_time_travellers: Option<TimeTravellingRegions>,
    mut op: impl FnMut(Location, Option<&DenseBitSet<RegionVid>>, bool),
) {
    // Forwards:
    if location.statement_index < block_data.statements.len() {
        let successor_location = location.successor_within_block();
        let time_travellers = maybe_time_travellers.as_ref().and_then(|t| t.to_next_loc.as_ref());
        op(successor_location, time_travellers, true);
    } else {
        for successor_block in block_data.terminator().successors() {
            let successor_location = Location { block: successor_block, statement_index: 0 };
            let time_travellers = maybe_time_travellers
                .as_ref()
                .and_then(|t| t.to_successor_blocks.as_ref().and_then(|x| x.row(successor_block)));
            op(successor_location, time_travellers, true);
        }
    }

    // Backwards:
    if location.statement_index > 0 {
        let predecessor_location = location.predecessor_within_block();
        let time_travellers = maybe_time_travellers.as_ref().and_then(|t| t.to_prev_stmt.as_ref());
        op(predecessor_location, time_travellers, false);
    } else {
        for &predecessor_block in &pcx.body.basic_blocks.predecessors()[location.block] {
            let predecessor_location = Location {
                block: predecessor_block,
                statement_index: pcx.body[predecessor_block].statements.len(),
            };
            let time_travellers = maybe_time_travellers.as_ref().and_then(|t| {
                t.to_predecessor_blocks.as_ref().and_then(|x| x.row(predecessor_block))
            });
            op(predecessor_location, time_travellers, false);
        }
    }
}
