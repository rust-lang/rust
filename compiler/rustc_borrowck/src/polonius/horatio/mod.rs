#![allow(dead_code)]
#![deny(unused_imports)]
mod constraints;
mod live_region_variance;
mod loan_invalidations;
mod location_sensitive;
mod polonius_block;
use std::cell::OnceCell;
use std::sync::LazyLock;

use constraints::Constraints;
use location_sensitive::LocationSensitiveAnalysis;
use polonius_block::PoloniusBlock;
use rustc_index::IndexVec;
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::{self, BasicBlock, Body, Local, Location, Place, Statement, Terminator};
use rustc_middle::ty::TyCtxt;
use rustc_mir_dataflow::points::DenseLocationMap;
use smallvec::{SmallVec, smallvec};

use super::ConstraintDirection;
use crate::{
    BorrowData, BorrowIndex, BorrowSet, PlaceConflictBias, PlaceExt, RegionInferenceContext,
    RegionVid, places_conflict,
};

/// This toggles the `my_println!` and `my_print!` macros. Those macros are used here and there to
/// print tracing information about Polonius.
pub(crate) const MY_DEBUG_PRINTS: LazyLock<bool> = LazyLock::new(|| {
    matches!(std::env::var("POLONIUS_TRACING").as_ref().map(String::as_str), Ok("1"))
});

macro_rules! my_println {
    ($($x:expr),*) => {
        if *crate::polonius::horatio::MY_DEBUG_PRINTS {
            println!($($x,)*);
        }
    };
}
pub(crate) use my_println;

macro_rules! my_print {
    ($($x:expr),*) => {
        if *crate::polonius::horatio::MY_DEBUG_PRINTS {
            print!($($x,)*);
        }
    };
}
pub(crate) use my_print;

/// A cache remembering whether a loan is killed at a block.
type KillsCache = IndexVec<PoloniusBlock, Option<KillAtBlock>>;

/// The main struct of Polonius which computes if loans are active at certain locations.
pub(crate) struct Polonius<'a, 'tcx> {
    pub pcx: PoloniusContext<'a, 'tcx>,
    borrows: IndexVec<BorrowIndex, Option<PoloniusBorrowData>>,
}

pub(crate) struct PoloniusContext<'a, 'tcx> {
    /// A cache that is only computed if we need the location sensitive analysis.
    cache: OnceCell<Cache<'a, 'tcx>>,

    /// For every block, we store a set of all proceeding blocks.
    ///
    /// ```
    ///       a
    ///      / \
    ///     b   c
    ///      \ /
    ///       d
    /// ```
    /// In this case we have:
    /// ```
    /// a: {}
    /// b: {a}
    /// c: {a}
    /// d: {a, b, c}
    /// ```
    transitive_predecessors: IndexVec<BasicBlock, DenseBitSet<BasicBlock>>,

    /// For every block we store the immediate predecessors.
    ///
    /// ```text
    ///       a
    ///      / \
    ///     b   c
    ///      \ /
    ///       d
    /// ```
    /// In this case we have:
    /// ```
    /// a: {}
    /// b: {a}
    /// c: {a}
    /// d: {b, c}
    /// ```
    // FIXME: This is equivalent to `BasicBlocks.predecessors` but uses bit sets instead of
    // `SmallVec`. Maybe that should be replaced by this.
    adjacent_predecessors: IndexVec<BasicBlock, DenseBitSet<BasicBlock>>,

    /// Only computed for diagnostics: The regions that outlive free regions are used to distinguish
    /// relevant live locals from boring locals. A boring local is one whose type contains only such
    /// regions. Polonius currently has more boring locals than NLLs so we record the latter to use
    /// in errors and diagnostics, to focus on the locals we consider relevant and match NLL
    /// diagnostics.
    boring_nll_locals: OnceCell<DenseBitSet<Local>>,

    tcx: TyCtxt<'tcx>,
    regioncx: &'a RegionInferenceContext<'tcx>,
    body: &'a Body<'tcx>,
    location_map: &'a DenseLocationMap,
    borrow_set: &'a BorrowSet<'tcx>,
}

struct Cache<'a, 'tcx> {
    /// All universal regions.
    universal_regions: DenseBitSet<RegionVid>,

    /// All outlives constraints.
    constraints: Constraints<'a, 'tcx>,
}

/// A borrow with some context.
///
/// It turns out that the index operation on [`BorrowSet`] takes a fair bit of time if executed many
/// many times. So we want to keep a reference to the [`BorrowData`] as well. The problem is that
/// sometimes a [`BorrowIndex`] is required, sometimes a `&[BorrowData]`, and sometimes we need a
/// reference to the [`Body`] or something else. So we bundles all this information in one struct.
#[derive(Copy, Clone)]
struct BorrowContext<'a, 'b, 'tcx> {
    pcx: &'a PoloniusContext<'b, 'tcx>,
    borrow_idx: BorrowIndex,
    borrow: &'a BorrowData<'tcx>,
}

/// Data used when computing when a loan is active.
enum PoloniusBorrowData {
    /// This borrow should be ignored.
    Ignored,

    Data {
        /// A cache of kills for this loan.
        kills_cache: KillsCache,
        scope_computation: Option<LocationSensitiveAnalysis>,
    },
}

/// Information of when/if a loan is killed at a block.
#[derive(Debug, Copy, Clone)]
enum KillAtBlock {
    /// The loan is not killed at this block.
    NotKilled,

    /// The loan is killed.
    Killed { statement_index: usize },
}
use KillAtBlock::*;

impl<'a, 'tcx> PoloniusContext<'a, 'tcx> {
    pub(crate) fn new(
        tcx: TyCtxt<'tcx>,
        regioncx: &'a RegionInferenceContext<'tcx>,
        body: &'a Body<'tcx>,
        location_map: &'a DenseLocationMap,
        borrow_set: &'a BorrowSet<'tcx>,
    ) -> Self {
        // Compute `transitive_predecessors` and `adjacent_predecessors`.
        let mut transitive_predecessors = IndexVec::from_elem_n(
            DenseBitSet::new_empty(body.basic_blocks.len()),
            body.basic_blocks.len(),
        );
        let mut adjacent_predecessors = transitive_predecessors.clone();
        // The stack is initially a reversed postorder traversal of the CFG. However, we might add
        // add blocks again to the stack if we have loops.
        let mut stack =
            body.basic_blocks.reverse_postorder().iter().rev().copied().collect::<Vec<_>>();
        // We keep track of all blocks that are currently not in the stack.
        let mut not_in_stack = DenseBitSet::new_empty(body.basic_blocks.len());
        while let Some(block) = stack.pop() {
            not_in_stack.insert(block);

            // Loop over all successors to the block and add `block` to their predecessors.
            for succ_block in body.basic_blocks[block].terminator().successors() {
                // Keep track of whether the transitive predecessors of `succ_block` has changed.
                let mut changed = false;

                // Insert `block` in `succ_block`s predecessors.
                if adjacent_predecessors[succ_block].insert(block) {
                    // Remember that `adjacent_predecessors` is a subset of
                    // `transitive_predecessors`.
                    changed |= transitive_predecessors[succ_block].insert(block);
                }

                // Add all transitive predecessors of `block` to the transitive predecessors of
                // `succ_block`.
                if block != succ_block {
                    let (blocks_predecessors, succ_blocks_predecessors) =
                        transitive_predecessors.pick2_mut(block, succ_block);
                    changed |= succ_blocks_predecessors.union(blocks_predecessors);

                    // Check if the `succ_block`s transitive predecessors changed. If so, we may
                    // need to add it to the stack again.
                    if changed && not_in_stack.remove(succ_block) {
                        stack.push(succ_block);
                    }
                }
            }

            debug_assert!(transitive_predecessors[block].superset(&adjacent_predecessors[block]));
        }

        Self {
            cache: OnceCell::new(),
            transitive_predecessors,
            adjacent_predecessors,
            boring_nll_locals: OnceCell::new(),
            tcx,
            regioncx,
            body,
            location_map,
            borrow_set,
        }
    }

    fn cache(&self) -> &Cache<'a, 'tcx> {
        self.cache.get_or_init(|| {
            let mut universal_regions = DenseBitSet::new_empty(self.regioncx.num_regions());
            universal_regions
                .insert_range(self.regioncx.universal_regions().universal_regions_range());

            let mut constraints =
                Constraints::new(self.tcx, self.regioncx, self.body, self.location_map);
            for constraint in self.regioncx.outlives_constraints() {
                constraints.add_constraint(&constraint);
            }

            Cache { universal_regions, constraints }
        })
    }

    fn boring_nll_locals(&self) -> &DenseBitSet<Local> {
        self.boring_nll_locals.get_or_init(|| {
            let mut free_regions = DenseBitSet::new_empty(self.regioncx.num_regions());
            for region in self.regioncx.universal_regions().universal_regions_iter() {
                free_regions.insert(region);
            }
            self.cache().constraints.add_dependent_regions_reversed(&mut free_regions);

            let mut boring_locals = DenseBitSet::new_empty(self.body.local_decls.len());
            for (local, local_decl) in self.body.local_decls.iter_enumerated() {
                if self
                    .tcx
                    .all_free_regions_meet(&local_decl.ty, |r| free_regions.contains(r.as_var()))
                {
                    boring_locals.insert(local);
                }
            }

            boring_locals
        })
    }

    pub(crate) fn is_boring_local(&self, local: Local) -> bool {
        self.boring_nll_locals().contains(local)
    }

    /// Returns `true` iff `a` is earlier in the control flow graph than `b`.
    #[inline]
    fn is_predecessor(&self, a: Location, b: Location) -> bool {
        a.block == b.block && a.statement_index < b.statement_index
            || self.transitive_predecessors[b.block].contains(a.block)
    }
}

impl<'a, 'b, 'tcx> BorrowContext<'a, 'b, 'tcx> {
    /// Construct a new empty set with capacity for [`PoloniusBlock`]s.
    fn new_polonius_block_set(self) -> DenseBitSet<PoloniusBlock> {
        DenseBitSet::new_empty(PoloniusBlock::num_blocks(self))
    }

    fn dependent_regions(&self) -> &DenseBitSet<RegionVid> {
        self.borrow.dependent_regions.get_or_init(|| {
            let mut dependent_regions = DenseBitSet::new_empty(self.pcx.regioncx.num_regions());
            dependent_regions.insert(self.borrow.region);
            self.pcx.cache().constraints.add_dependent_regions(&mut dependent_regions);
            dependent_regions
        })
    }

    fn has_live_region_at(&self, location: Location) -> bool {
        self.pcx.regioncx.region_contains(self.borrow.region, location)
    }
}

impl<'a, 'tcx> Polonius<'a, 'tcx> {
    pub(crate) fn new(
        tcx: TyCtxt<'tcx>,
        regioncx: &'a RegionInferenceContext<'tcx>,
        body: &'a Body<'tcx>,
        location_map: &'a DenseLocationMap,
        borrow_set: &'a BorrowSet<'tcx>,
    ) -> Self {
        Self {
            pcx: PoloniusContext::new(tcx, regioncx, body, location_map, borrow_set),
            borrows: IndexVec::new(),
        }
    }

    /// Quick check to check if a loan is active at a certain point in the CFG.
    ///
    /// If this function returns `false`, we know for sure that the loan is not active at
    /// `location`, otherwise it may or may not be active.
    ///
    /// The purpose of this function is to be really quick. In most cases it will return `false` and
    /// no conflict is therefore possible. In the rare situations it returns `true`, the caller
    /// should proceed with other more time consuming methods of checking for a conflict and
    /// eventually call the [`Polonius::loan_is_active`] function which will give a definite answer.
    #[inline]
    pub(crate) fn loan_maybe_active_at(
        &mut self,
        borrow_idx: BorrowIndex,
        borrow: &BorrowData<'tcx>,
        location: Location,
    ) -> bool {
        // Check if this location can never be reached by the borrow.
        if !self.pcx.is_predecessor(borrow.reserve_location(), location) {
            return false;
        }

        let bcx = BorrowContext { pcx: &self.pcx, borrow_idx, borrow };

        if !bcx.has_live_region_at(location) {
            return false;
        }

        true
    }

    /// Check if a loan is is active at a point in the CFG.
    pub(crate) fn loan_is_active_at(
        &mut self,
        borrow_idx: BorrowIndex,
        borrow: &BorrowData<'tcx>,
        location: Location,
    ) -> bool {
        let maybe_borrow_data = self.borrows.ensure_contains_elem(borrow_idx, || None);
        let (kills_cache, scope_computation) = match maybe_borrow_data {
            Some(PoloniusBorrowData::Ignored) => return false,
            Some(PoloniusBorrowData::Data { scope_computation, kills_cache }) => {
                if let Some(scope_computation) = &scope_computation {
                    // Check if we have already computed an "in scope-value" for location.
                    if scope_computation.is_finished {
                        // If the scope computation is finished, it's appropriate to return `false` if no
                        // node for the location exists.
                        return scope_computation.nodes.get(&location).is_some_and(|x| x.is_active);

                        // If the computation is not finished, we can only be sure if the `in_scope`-field
                        // has been set to `true` for the relevant node.
                    } else if scope_computation.nodes.get(&location).is_some_and(|x| x.is_active) {
                        return true;
                    }
                }

                (kills_cache, scope_computation)
            }
            None => {
                // Check if this borrow is ignored.
                if borrow.borrowed_place().ignore_borrow(
                    self.pcx.tcx,
                    self.pcx.body,
                    &self.pcx.borrow_set.locals_state_at_exit,
                ) {
                    *maybe_borrow_data = Some(PoloniusBorrowData::Ignored);
                    return false;
                }

                *maybe_borrow_data = Some(PoloniusBorrowData::Data {
                    kills_cache: IndexVec::new(),
                    scope_computation: None,
                });

                let Some(PoloniusBorrowData::Data { kills_cache, scope_computation }) =
                    maybe_borrow_data
                else {
                    unreachable!()
                };
                (kills_cache, scope_computation)
            }
        };

        let bcx = BorrowContext { pcx: &self.pcx, borrow_idx, borrow };

        // Check if the loan is killed anywhere between its reserve location and `location`.
        let Some(live_paths) = live_paths(bcx, kills_cache, location) else {
            return false;
        };

        if self.pcx.tcx.sess.opts.unstable_opts.polonius.is_next_enabled() {
            scope_computation.get_or_insert_with(|| LocationSensitiveAnalysis::new(bcx)).compute(
                bcx,
                kills_cache,
                location,
                live_paths,
            )
        } else {
            true
        }
    }
}

/// Returns `true` if the loan is killed at `location`. Note that the kill takes effect at the next
/// statement.
fn is_killed(
    bcx: BorrowContext<'_, '_, '_>,
    kills_cache: &mut KillsCache,
    location: Location,
) -> bool {
    let polonius_block = PoloniusBlock::from_location(bcx, location);

    // Check if we already know the answer.
    match kills_cache.get(polonius_block) {
        Some(Some(Killed { statement_index })) => {
            return *statement_index == location.statement_index;
        }
        Some(Some(NotKilled)) => return false,
        Some(None) | None => (),
    }
    // The answer was not known so we have to compute it ourselfs.

    let is_kill = !bcx.has_live_region_at(location)
        || if let Some(stmt) = bcx.pcx.body[location.block].statements.get(location.statement_index)
        {
            is_killed_at_stmt(bcx, stmt)
        } else {
            is_killed_at_terminator(bcx, &bcx.pcx.body[location.block].terminator())
        };

    // If we had a kill at this location, we should add it to the cache.
    if is_kill {
        *kills_cache.ensure_contains_elem(polonius_block, || None) =
            Some(Killed { statement_index: location.statement_index });
    }

    is_kill
}

/// Calculate when/if a loan goes out of scope for a set of statements in a block.
fn is_killed_at_block(
    bcx: BorrowContext<'_, '_, '_>,
    kills_cache: &mut KillsCache,
    block: PoloniusBlock,
) -> bool {
    let res = kills_cache.get_or_insert_with(block, || {
        let block_data = &bcx.pcx.body[block.basic_block(bcx)];
        for statement_index in block.first_index(bcx)..=block.last_index(bcx) {
            let location = Location { statement_index, block: block.basic_block(bcx) };

            let is_kill = !bcx.has_live_region_at(location)
                || if let Some(stmt) = block_data.statements.get(statement_index) {
                    is_killed_at_stmt(bcx, stmt)
                } else {
                    is_killed_at_terminator(bcx, &block_data.terminator())
                };

            if is_kill {
                return Killed { statement_index };
            }
        }

        NotKilled
    });

    matches!(res, Killed { .. })
}

/// Given that the borrow was in scope on entry to this statement, check if it goes out of scope
/// till the next location.
#[inline]
fn is_killed_at_stmt<'tcx>(bcx: BorrowContext<'_, '_, 'tcx>, stmt: &Statement<'tcx>) -> bool {
    match &stmt.kind {
        mir::StatementKind::Assign(box (lhs, _rhs)) => kill_on_place(bcx, *lhs),
        mir::StatementKind::StorageDead(local) => {
            bcx.pcx.borrow_set.local_map.get(local).is_some_and(|bs| bs.contains(&bcx.borrow_idx))
        }
        _ => false,
    }
}

/// Given that the borrow was in scope on entry to this terminator, check if it goes out of scope
/// till the succeeding blocks.
#[inline]
fn is_killed_at_terminator<'tcx>(
    bcx: BorrowContext<'_, '_, 'tcx>,
    terminator: &Terminator<'tcx>,
) -> bool {
    match &terminator.kind {
        // A `Call` terminator's return value can be a local which has borrows, so we need to record
        // those as killed as well.
        mir::TerminatorKind::Call { destination, .. } => kill_on_place(bcx, *destination),
        mir::TerminatorKind::InlineAsm { operands, .. } => operands.iter().any(|op| {
            if let mir::InlineAsmOperand::Out { place: Some(place), .. }
            | mir::InlineAsmOperand::InOut { out_place: Some(place), .. } = op
            {
                kill_on_place(bcx, *place)
            } else {
                false
            }
        }),
        _ => false,
    }
}

#[inline]
fn kill_on_place<'tcx>(bcx: BorrowContext<'_, '_, 'tcx>, place: Place<'tcx>) -> bool {
    bcx.pcx.borrow_set.local_map.get(&place.local).is_some_and(|bs| bs.contains(&bcx.borrow_idx))
        && if place.projection.is_empty() {
            !bcx.pcx.body.local_decls[place.local].is_ref_to_static()
        } else {
            places_conflict(
                bcx.pcx.tcx,
                bcx.pcx.body,
                bcx.borrow.borrowed_place,
                place,
                PlaceConflictBias::NoOverlap,
            )
        }
}

#[inline(never)] // FIXME: Remove this.
fn live_paths(
    bcx: BorrowContext<'_, '_, '_>,
    kills_cache: &mut KillsCache,
    destination: Location,
) -> Option<DenseBitSet<PoloniusBlock>> {
    // `destination_block` is the `PoloniusBlock` for `destination`.
    let destination_block = PoloniusBlock::from_location(bcx, destination);

    // We begin by checking the relevant statements in `destination_block`.
    // FIXME: Is this the most efficient solution?
    for statement_index in destination_block.first_index(bcx)..destination.statement_index {
        let location = Location { block: destination.block, statement_index };
        if is_killed(bcx, kills_cache, location) {
            return None;
        }
    }

    if destination_block.is_introduction_block(bcx) {
        // We are finished.
        return Some(bcx.new_polonius_block_set());
    }

    // Traverse all blocks between `reserve_location` and `destination` in the CFG and check for
    // kills. If there is no live path from `reserve_location` to `destination`, we no for sure
    // that the loan is dead at `destination`.

    // Keep track of all visited `PoloniusBlock`s.
    let mut visited = bcx.new_polonius_block_set();

    // The stack contains `(block, path)` pairs, where `block` is a `PoloniusBlock` and `path` is
    // a set of `PoloniusBlock`s making a path from `reserve_location` to `destination_block`.
    // In this way we can record the live paths.
    let introduction_block = PoloniusBlock::introduction_block(bcx);
    let mut stack: SmallVec<[(PoloniusBlock, DenseBitSet<PoloniusBlock>); 4]> =
        smallvec![(introduction_block, bcx.new_polonius_block_set())];
    visited.insert(introduction_block);

    let mut valid_paths = None;

    while let Some((block, path)) = stack.pop() {
        // Check if the loan is killed in this block.
        if is_killed_at_block(bcx, kills_cache, block) {
            continue;
        }

        // Loop through all successors to `block` and follow those that are predecessors to
        // `destination.block`.
        for successor in block.successors(bcx) {
            let successor_bb = successor.basic_block(bcx);

            if successor == destination_block {
                // We have reached the destination so let's save this path.
                valid_paths.get_or_insert_with(|| bcx.new_polonius_block_set()).union(&path);

                // We continue traversal to record all live paths.
                continue;
            }

            if !visited.insert(successor) {
                continue;
            }

            // Check that `successor` is a predecessor of `destination_block`.
            //
            // Given two `PoloniusBlock`s a and b, then a is a predecessor of b iff
            // `a.basic_block()` is a predecessor of `b.basic_block()`, or a is the "before
            // introduction block" and b is the "introduction block".
            if !bcx.pcx.transitive_predecessors[destination.block].contains(successor_bb)
                || destination_block.is_introduction_block(bcx)
                    && successor.is_before_introduction_block(bcx)
            {
                // `successor` is not a predecessor of `destination_block`.
                continue;
            }

            // Push `successor` to `path`.
            let mut path = path.clone();
            path.insert(successor);
            stack.push((successor, path));
        }
    }

    valid_paths
}

/// Remove dead regions from the set of associated regions.
fn remove_dead_regions(
    pcx: &PoloniusContext<'_, '_>,
    location: Location,
    region_set: &mut DenseBitSet<RegionVid>,
) {
    for region in region_set.clone().iter() {
        if !pcx.regioncx.liveness_constraints().is_live_at(region, location) {
            region_set.remove(region);
        }
    }
}

/// FIXME: Just for debugging.
pub(crate) fn format_body_with_borrows<'tcx>(
    body: &Body<'tcx>,
    borrow_set: &BorrowSet<'tcx>,
) -> String {
    let mut res = String::default();
    for (block, block_data) in body.basic_blocks.iter_enumerated() {
        res += &format!("{:?}:\n", block);
        for statement_index in 0..=block_data.statements.len() {
            let location = Location { block, statement_index };
            res += &format!("  {}: ", statement_index);
            if let Some(stmt) = body[location.block].statements.get(location.statement_index) {
                res += &format!("{:?}\n", stmt);
            } else {
                debug_assert_eq!(location.statement_index, body[location.block].statements.len());
                let terminator = body[location.block].terminator();
                res += &format!("{:?}\n", terminator.kind);
            }

            let introduced_borrows = borrow_set
                .iter_enumerated()
                .filter(|(_, b)| b.reserve_location == location)
                .collect::<Vec<_>>();
            if !introduced_borrows.is_empty() {
                res += "    reserved borrows: ";
                for (borrow_idx, _) in introduced_borrows {
                    res += &format!("{:?}, ", borrow_idx);
                }
                res += "\n"
            }
        }
    }
    res
}
