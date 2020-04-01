//! Random access inspection of the results of a dataflow analysis.

use std::borrow::Borrow;

use rustc_index::bit_set::BitSet;
use rustc_middle::mir::{self, BasicBlock, Location, TerminatorKind};

use super::{Analysis, Results};

/// A `ResultsCursor` that borrows the underlying `Results`.
pub type ResultsRefCursor<'a, 'mir, 'tcx, A> = ResultsCursor<'mir, 'tcx, A, &'a Results<'tcx, A>>;

/// Allows random access inspection of the results of a dataflow analysis.
///
/// This cursor only has linear performance within a basic block when its statements are visited in
/// order. In the worst case—when statements are visited in *reverse* order—performance will be
/// quadratic in the number of statements in the block. The order in which basic blocks are
/// inspected has no impact on performance.
///
/// A `ResultsCursor` can either own (the default) or borrow the dataflow results it inspects. The
/// type of ownership is determined by `R` (see `ResultsRefCursor` above).
pub struct ResultsCursor<'mir, 'tcx, A, R = Results<'tcx, A>>
where
    A: Analysis<'tcx>,
{
    body: &'mir mir::Body<'tcx>,
    results: R,
    state: BitSet<A::Idx>,

    pos: CursorPosition,

    /// When this flag is set, the cursor is pointing at a `Call` or `Yield` terminator whose call
    /// return or resume effect has been applied to `state`.
    ///
    /// This flag helps to ensure that multiple calls to `seek_after_assume_success` with the
    /// same target will result in exactly one invocation of `apply_call_return_effect`. It is
    /// sufficient to clear this only in `seek_to_block_start`, since seeking away from a
    /// terminator will always require a cursor reset.
    success_effect_applied: bool,
}

impl<'mir, 'tcx, A, R> ResultsCursor<'mir, 'tcx, A, R>
where
    A: Analysis<'tcx>,
    R: Borrow<Results<'tcx, A>>,
{
    /// Returns a new cursor for `results` that points to the start of the `START_BLOCK`.
    pub fn new(body: &'mir mir::Body<'tcx>, results: R) -> Self {
        ResultsCursor {
            body,
            pos: CursorPosition::BlockStart(mir::START_BLOCK),
            state: results.borrow().entry_sets[mir::START_BLOCK].clone(),
            success_effect_applied: false,
            results,
        }
    }

    /// Returns the `Analysis` used to generate the underlying results.
    pub fn analysis(&self) -> &A {
        &self.results.borrow().analysis
    }

    /// Returns the dataflow state at the current location.
    pub fn get(&self) -> &BitSet<A::Idx> {
        &self.state
    }

    /// Returns `true` if the dataflow state at the current location contains the given element.
    ///
    /// Shorthand for `self.get().contains(elem)`
    pub fn contains(&self, elem: A::Idx) -> bool {
        self.state.contains(elem)
    }

    /// Resets the cursor to the start of the given basic block.
    pub fn seek_to_block_start(&mut self, block: BasicBlock) {
        self.state.overwrite(&self.results.borrow().entry_sets[block]);
        self.pos = CursorPosition::BlockStart(block);
        self.success_effect_applied = false;
    }

    /// Advances the cursor to hold all effects up to and including to the "before" effect of the
    /// statement (or terminator) at the given location.
    ///
    /// If you wish to observe the full effect of a statement or terminator, not just the "before"
    /// effect, use `seek_after` or `seek_after_assume_success`.
    pub fn seek_before(&mut self, target: Location) {
        assert!(target <= self.body.terminator_loc(target.block));
        self.seek_(target, false);
    }

    /// Advances the cursor to hold the full effect of all statements (and possibly closing
    /// terminators) up to and including the `target`.
    ///
    /// If the `target` is a `Call` terminator, any call return effect for that terminator will
    /// **not** be observed. Use `seek_after_assume_success` if you wish to observe the call
    /// return effect.
    pub fn seek_after(&mut self, target: Location) {
        assert!(target <= self.body.terminator_loc(target.block));

        // If we have already applied the call return effect, we are currently pointing at a `Call`
        // terminator. Unconditionally reset the dataflow cursor, since there is no way to "undo"
        // the call return effect.
        if self.success_effect_applied {
            self.seek_to_block_start(target.block);
        }

        self.seek_(target, true);
    }

    /// Advances the cursor to hold all effects up to and including of the statement (or
    /// terminator) at the given location.
    ///
    /// If the `target` is a `Call` or `Yield` terminator, any call return or resume effect for that
    /// terminator will be observed. Use `seek_after` if you do **not** wish to observe the
    /// "success" effect.
    pub fn seek_after_assume_success(&mut self, target: Location) {
        let terminator_loc = self.body.terminator_loc(target.block);
        assert!(target.statement_index <= terminator_loc.statement_index);

        self.seek_(target, true);

        if target != terminator_loc || self.success_effect_applied {
            return;
        }

        // Apply the effect of the "success" path of the terminator.

        self.success_effect_applied = true;
        let terminator = self.body.basic_blocks()[target.block].terminator();
        match &terminator.kind {
            TerminatorKind::Call { destination: Some((return_place, _)), func, args, .. } => {
                self.results.borrow().analysis.apply_call_return_effect(
                    &mut self.state,
                    target.block,
                    func,
                    args,
                    *return_place,
                );
            }
            TerminatorKind::Yield { resume, resume_arg, .. } => {
                self.results.borrow().analysis.apply_yield_resume_effect(
                    &mut self.state,
                    *resume,
                    *resume_arg,
                );
            }
            _ => {}
        }
    }

    fn seek_(&mut self, target: Location, apply_after_effect_at_target: bool) {
        use CursorPosition::*;

        match self.pos {
            // Return early if we are already at the target location.
            Before(curr) if curr == target && !apply_after_effect_at_target => return,
            After(curr) if curr == target && apply_after_effect_at_target => return,

            // Otherwise, we must reset to the start of the target block if...

            // we are in a different block entirely.
            BlockStart(block) | Before(Location { block, .. }) | After(Location { block, .. })
                if block != target.block =>
            {
                self.seek_to_block_start(target.block)
            }

            // we are in the same block but have advanced past the target statement.
            Before(curr) | After(curr) if curr.statement_index > target.statement_index => {
                self.seek_to_block_start(target.block)
            }

            // we have already applied the entire effect of a statement but only wish to observe
            // its "before" effect.
            After(curr)
                if curr.statement_index == target.statement_index
                    && !apply_after_effect_at_target =>
            {
                self.seek_to_block_start(target.block)
            }

            // N.B., `success_effect_applied` is checked in `seek_after`, not here.
            _ => (),
        }

        let analysis = &self.results.borrow().analysis;
        let block_data = &self.body.basic_blocks()[target.block];

        // At this point, the cursor is in the same block as the target location at an earlier
        // statement.
        debug_assert_eq!(target.block, self.pos.block());

        // Find the first statement whose transfer function has not yet been applied.
        let first_unapplied_statement = match self.pos {
            BlockStart(_) => 0,
            After(Location { statement_index, .. }) => statement_index + 1,

            // If we have only applied the "before" effect for the current statement, apply the
            // remainder before continuing.
            Before(curr) => {
                if curr.statement_index == block_data.statements.len() {
                    let terminator = block_data.terminator();
                    analysis.apply_terminator_effect(&mut self.state, terminator, curr);
                } else {
                    let statement = &block_data.statements[curr.statement_index];
                    analysis.apply_statement_effect(&mut self.state, statement, curr);
                }

                // If all we needed to do was go from `Before` to `After` in the same statement,
                // we are now done.
                if curr.statement_index == target.statement_index {
                    debug_assert!(apply_after_effect_at_target);
                    self.pos = After(target);
                    return;
                }

                curr.statement_index + 1
            }
        };

        // We have now applied all effects prior to `first_unapplied_statement`.

        // Apply the effects of all statements before `target`.
        let mut location = Location { block: target.block, statement_index: 0 };
        for statement_index in first_unapplied_statement..target.statement_index {
            location.statement_index = statement_index;
            let statement = &block_data.statements[statement_index];
            analysis.apply_before_statement_effect(&mut self.state, statement, location);
            analysis.apply_statement_effect(&mut self.state, statement, location);
        }

        // Apply the effect of the statement (or terminator) at `target`.
        location.statement_index = target.statement_index;
        if target.statement_index == block_data.statements.len() {
            let terminator = &block_data.terminator();
            analysis.apply_before_terminator_effect(&mut self.state, terminator, location);

            if apply_after_effect_at_target {
                analysis.apply_terminator_effect(&mut self.state, terminator, location);
                self.pos = After(target);
            } else {
                self.pos = Before(target);
            }
        } else {
            let statement = &block_data.statements[target.statement_index];
            analysis.apply_before_statement_effect(&mut self.state, statement, location);

            if apply_after_effect_at_target {
                analysis.apply_statement_effect(&mut self.state, statement, location);
                self.pos = After(target)
            } else {
                self.pos = Before(target);
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum CursorPosition {
    /// No effects within this block have been applied.
    BlockStart(BasicBlock),

    /// Only the "before" effect of the statement (or terminator) at this location has been
    /// applied (along with the effects of all previous statements).
    Before(Location),

    /// The effects of all statements up to and including the one at this location have been
    /// applied.
    After(Location),
}

impl CursorPosition {
    fn block(&self) -> BasicBlock {
        match *self {
            Self::BlockStart(block) => block,
            Self::Before(loc) | Self::After(loc) => loc.block,
        }
    }
}
