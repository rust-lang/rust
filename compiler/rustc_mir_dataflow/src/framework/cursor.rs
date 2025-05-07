//! Random access inspection of the results of a dataflow analysis.

use std::cmp::Ordering;
use std::ops::{Deref, DerefMut};

#[cfg(debug_assertions)]
use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::{self, BasicBlock, Location};

use super::{Analysis, Direction, Effect, EffectIndex, Results};

/// Some `ResultsCursor`s want to own a `Results`, and some want to borrow a `Results`, either
/// mutable or immutably. This type allows all of the above. It's similar to `Cow`.
pub enum ResultsHandle<'a, 'tcx, A>
where
    A: Analysis<'tcx>,
{
    BorrowedMut(&'a mut Results<'tcx, A>),
    Owned(Results<'tcx, A>),
}

impl<'tcx, A> Deref for ResultsHandle<'_, 'tcx, A>
where
    A: Analysis<'tcx>,
{
    type Target = Results<'tcx, A>;

    fn deref(&self) -> &Results<'tcx, A> {
        match self {
            ResultsHandle::BorrowedMut(borrowed) => borrowed,
            ResultsHandle::Owned(owned) => owned,
        }
    }
}

impl<'tcx, A> DerefMut for ResultsHandle<'_, 'tcx, A>
where
    A: Analysis<'tcx>,
{
    fn deref_mut(&mut self) -> &mut Results<'tcx, A> {
        match self {
            ResultsHandle::BorrowedMut(borrowed) => borrowed,
            ResultsHandle::Owned(owned) => owned,
        }
    }
}

/// Allows random access inspection of the results of a dataflow analysis. Use this when you want
/// to inspect domain values only in certain locations; use `ResultsVisitor` if you want to inspect
/// domain values in many or all locations.
///
/// Because `Results` only has domain values for the entry of each basic block, these inspections
/// involve some amount of domain value recomputations. This cursor only has linear performance
/// within a basic block when its statements are visited in the same order as the `DIRECTION` of
/// the analysis. In the worst case—when statements are visited in *reverse* order—performance will
/// be quadratic in the number of statements in the block. The order in which basic blocks are
/// inspected has no impact on performance.
pub struct ResultsCursor<'mir, 'tcx, A>
where
    A: Analysis<'tcx>,
{
    body: &'mir mir::Body<'tcx>,
    results: ResultsHandle<'mir, 'tcx, A>,
    state: A::Domain,

    pos: CursorPosition,

    /// Indicates that `state` has been modified with a custom effect.
    ///
    /// When this flag is set, we need to reset to an entry set before doing a seek.
    state_needs_reset: bool,

    #[cfg(debug_assertions)]
    reachable_blocks: DenseBitSet<BasicBlock>,
}

impl<'mir, 'tcx, A> ResultsCursor<'mir, 'tcx, A>
where
    A: Analysis<'tcx>,
{
    /// Returns the dataflow state at the current location.
    pub fn get(&self) -> &A::Domain {
        &self.state
    }

    /// Returns the body this analysis was run on.
    pub fn body(&self) -> &'mir mir::Body<'tcx> {
        self.body
    }

    /// Returns a new cursor that can inspect `results`.
    pub fn new(body: &'mir mir::Body<'tcx>, results: ResultsHandle<'mir, 'tcx, A>) -> Self {
        let bottom_value = results.analysis.bottom_value(body);
        ResultsCursor {
            body,
            results,

            // Initialize to the `bottom_value` and set `state_needs_reset` to tell the cursor that
            // it needs to reset to block entry before the first seek. The cursor position is
            // immaterial.
            state_needs_reset: true,
            state: bottom_value,
            pos: CursorPosition::block_entry(mir::START_BLOCK),

            #[cfg(debug_assertions)]
            reachable_blocks: mir::traversal::reachable_as_bitset(body),
        }
    }

    /// Allows inspection of unreachable basic blocks even with `debug_assertions` enabled.
    #[cfg(test)]
    pub(crate) fn allow_unreachable(&mut self) {
        #[cfg(debug_assertions)]
        self.reachable_blocks.insert_all()
    }

    /// Returns the `Analysis` used to generate the underlying `Results`.
    pub fn analysis(&self) -> &A {
        &self.results.analysis
    }

    /// Resets the cursor to hold the entry set for the given basic block.
    ///
    /// For forward dataflow analyses, this is the dataflow state prior to the first statement.
    ///
    /// For backward dataflow analyses, this is the dataflow state after the terminator.
    pub(super) fn seek_to_block_entry(&mut self, block: BasicBlock) {
        #[cfg(debug_assertions)]
        assert!(self.reachable_blocks.contains(block));

        self.state.clone_from(self.results.entry_set_for_block(block));
        self.pos = CursorPosition::block_entry(block);
        self.state_needs_reset = false;
    }

    /// Resets the cursor to hold the state prior to the first statement in a basic block.
    ///
    /// For forward analyses, this is the entry set for the given block.
    ///
    /// For backward analyses, this is the state that will be propagated to its
    /// predecessors (ignoring edge-specific effects).
    pub fn seek_to_block_start(&mut self, block: BasicBlock) {
        if A::Direction::IS_FORWARD {
            self.seek_to_block_entry(block)
        } else {
            self.seek_after(Location { block, statement_index: 0 }, Effect::Primary)
        }
    }

    /// Resets the cursor to hold the state after the terminator in a basic block.
    ///
    /// For backward analyses, this is the entry set for the given block.
    ///
    /// For forward analyses, this is the state that will be propagated to its
    /// successors (ignoring edge-specific effects).
    pub fn seek_to_block_end(&mut self, block: BasicBlock) {
        if A::Direction::IS_BACKWARD {
            self.seek_to_block_entry(block)
        } else {
            self.seek_after(self.body.terminator_loc(block), Effect::Primary)
        }
    }

    /// Advances the cursor to hold the dataflow state at `target` before its "primary" effect is
    /// applied.
    ///
    /// The "early" effect at the target location *will be* applied.
    pub fn seek_before_primary_effect(&mut self, target: Location) {
        self.seek_after(target, Effect::Early)
    }

    /// Advances the cursor to hold the dataflow state at `target` after its "primary" effect is
    /// applied.
    ///
    /// The "early" effect at the target location will be applied as well.
    pub fn seek_after_primary_effect(&mut self, target: Location) {
        self.seek_after(target, Effect::Primary)
    }

    fn seek_after(&mut self, target: Location, effect: Effect) {
        assert!(target <= self.body.terminator_loc(target.block));

        // Reset to the entry of the target block if any of the following are true:
        //   - A custom effect has been applied to the cursor state.
        //   - We are in a different block than the target.
        //   - We are in the same block but have advanced past the target effect.
        if self.state_needs_reset || self.pos.block != target.block {
            self.seek_to_block_entry(target.block);
        } else if let Some(curr_effect) = self.pos.curr_effect_index {
            let mut ord = curr_effect.statement_index.cmp(&target.statement_index);
            if A::Direction::IS_BACKWARD {
                ord = ord.reverse()
            }

            match ord.then_with(|| curr_effect.effect.cmp(&effect)) {
                Ordering::Equal => return,
                Ordering::Greater => self.seek_to_block_entry(target.block),
                Ordering::Less => {}
            }
        }

        // At this point, the cursor is in the same block as the target location at an earlier
        // statement.
        debug_assert_eq!(target.block, self.pos.block);

        let block_data = &self.body[target.block];
        #[rustfmt::skip]
        let next_effect = if A::Direction::IS_FORWARD {
            self.pos.curr_effect_index.map_or_else(
                || Effect::Early.at_index(0),
                EffectIndex::next_in_forward_order,
            )
        } else {
            self.pos.curr_effect_index.map_or_else(
                || Effect::Early.at_index(block_data.statements.len()),
                EffectIndex::next_in_backward_order,
            )
        };

        let target_effect_index = effect.at_index(target.statement_index);

        A::Direction::apply_effects_in_range(
            &mut self.results.analysis,
            &mut self.state,
            target.block,
            block_data,
            next_effect..=target_effect_index,
        );

        self.pos =
            CursorPosition { block: target.block, curr_effect_index: Some(target_effect_index) };
    }

    /// Applies `f` to the cursor's internal state.
    ///
    /// This can be used, e.g., to apply the call return effect directly to the cursor without
    /// creating an extra copy of the dataflow state.
    pub fn apply_custom_effect(&mut self, f: impl FnOnce(&mut A, &mut A::Domain)) {
        f(&mut self.results.analysis, &mut self.state);
        self.state_needs_reset = true;
    }
}

#[derive(Clone, Copy, Debug)]
struct CursorPosition {
    block: BasicBlock,
    curr_effect_index: Option<EffectIndex>,
}

impl CursorPosition {
    fn block_entry(block: BasicBlock) -> CursorPosition {
        CursorPosition { block, curr_effect_index: None }
    }
}
