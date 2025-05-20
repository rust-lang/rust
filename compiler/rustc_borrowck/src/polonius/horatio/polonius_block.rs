use itertools::Either;
use rustc_middle::mir::{BasicBlock, Location};

use super::BorrowContext;

rustc_index::newtype_index! {
    /// A `PoloniusBlock` is a `BasicBlock` which splits the block where a loan is introduced into
    /// two blocks.
    ///
    /// The problem is that we want to record at most one location per block where a loan is killed.
    /// But a loan might be killed twice in the block where it is introduced, both before and after
    /// the reserve location. So we use an additional index to denote the introduction block up to
    /// and including the statement where the loan is introduced. This has the consequence that a
    /// `PoloniusBlock` is specific for a given loan.
    ///
    /// We call the block containing all statements after the reserve location for the
    /// "introduction block", and the block containing statements up to and including the reserve
    /// location "before introduction block". These names might be bad, but my (Tage's) fantacy
    /// struggles to come up with anything better.
    ///
    /// So if the loan is introduced at `bb2[2]`, `bb2[0..=2]` is the "before introduction block"
    /// and `bb2[3..]` is the "introduction block".
    ///
    /// For a given loan `l` introduced at a basic block `b`, a `PoloniusBlock` is equivalent to a
    /// `BasicBlock` with the following exceptions:
    /// - `PoloniusBlock::from_u32(b.as_u32())` is `l`'s introduction block.
    /// - `PoloniusBlock::from_usize(basic_blocks.len())` is `l`'s "before introduction block".
    #[debug_format = "pbb{}"]
    pub(super) struct PoloniusBlock {}
}

impl PoloniusBlock {
    /// Converts a [`BasicBlock`] to a [`PoloniusBlock`] assuming this is not the "before
    /// introduction block".
    #[inline]
    pub(super) fn from_basic_block(basic_block: BasicBlock) -> Self {
        Self::from_u32(basic_block.as_u32())
    }

    /// Get the "introduction block". I.E the first block where the loan is introduced.
    #[inline]
    pub(super) fn introduction_block(bcx: BorrowContext<'_, '_, '_>) -> Self {
        Self::from_basic_block(bcx.borrow.reserve_location.block)
    }

    /// Get the "before introduction block". I.E the block consisting of statements up to and
    /// including the loan's reserve location.
    #[inline]
    pub(super) fn before_introduction_block(bcx: BorrowContext<'_, '_, '_>) -> Self {
        Self::from_usize(bcx.pcx.body.basic_blocks.len())
    }

    /// Get the correct block from a loan and a location.
    #[inline]
    pub(super) fn from_location(bcx: BorrowContext<'_, '_, '_>, location: Location) -> Self {
        if location.block == bcx.borrow.reserve_location.block
            && location.statement_index <= bcx.borrow.reserve_location.statement_index
        {
            Self::before_introduction_block(bcx)
        } else {
            Self::from_basic_block(location.block)
        }
    }

    /// Returns the number of polonius blocks. THat is, the number of blocks + 1.
    #[inline]
    pub(super) fn num_blocks(bcx: BorrowContext<'_, '_, '_>) -> usize {
        bcx.pcx.body.basic_blocks.len() + 1
    }

    /// Get the [`BasicBlock`] containing this [`PoloniusBlock`].
    #[inline]
    pub(super) fn basic_block(self, bcx: BorrowContext<'_, '_, '_>) -> BasicBlock {
        if self.as_usize() == bcx.pcx.body.basic_blocks.len() {
            bcx.borrow.reserve_location.block
        } else {
            BasicBlock::from_u32(self.as_u32())
        }
    }

    /// Check if this is the "introduction block". I.E the block immediately after the loan has been
    /// introduced.
    #[inline]
    pub(super) fn is_introduction_block(self, bcx: BorrowContext<'_, '_, '_>) -> bool {
        self.as_u32() == bcx.borrow.reserve_location.block.as_u32()
    }

    /// Check if this is the "before introduction block". I.E the block containing statements up to
    /// and including the loan's reserve location.
    #[inline]
    pub(super) fn is_before_introduction_block(self, bcx: BorrowContext<'_, '_, '_>) -> bool {
        self.as_usize() == bcx.pcx.body.basic_blocks.len()
    }

    /// Get the index of the first statement in this block. This will be 0 except for the
    /// introduction block.
    #[inline]
    pub(super) fn first_index(self, bcx: BorrowContext<'_, '_, '_>) -> usize {
        if self.is_introduction_block(bcx) {
            bcx.borrow.reserve_location.statement_index + 1
        } else {
            0
        }
    }

    /// Get the last statement index for this block. For all blocks except the "before introduction
    /// block", this will point to a terminator, not a statement.
    #[inline]
    pub(super) fn last_index(self, bcx: BorrowContext<'_, '_, '_>) -> usize {
        if !self.is_before_introduction_block(bcx) {
            bcx.pcx.body.basic_blocks[self.basic_block(bcx)].statements.len()
        } else {
            bcx.borrow.reserve_location.statement_index
        }
    }

    /// Iterate over the successor blocks to this block.
    ///
    /// Note that this is same as [`Terminator::successors`] except for the "before introduction
    /// block" where it is the "introduction block".
    #[inline]
    pub(super) fn successors(
        self,
        bcx: BorrowContext<'_, '_, '_>,
    ) -> impl DoubleEndedIterator<Item = PoloniusBlock> {
        if !self.is_before_introduction_block(bcx) {
            Either::Left(bcx.pcx.body[self.basic_block(bcx)].terminator().successors().map(
                move |bb| {
                    if bb == bcx.borrow.reserve_location.block {
                        Self::before_introduction_block(bcx)
                    } else {
                        Self::from_basic_block(bb)
                    }
                },
            ))
        } else {
            Either::Right([Self::introduction_block(bcx)].into_iter())
        }
    }
}
