use rustc_index::{Idx, IndexVec};
use rustc_middle::mir::{BasicBlock, Body, Location};

/// Maps between a `Location` and a `PointIndex` (and vice versa).
pub struct DenseLocationMap {
    /// For each basic block, how many points are contained within?
    statements_before_block: IndexVec<BasicBlock, usize>,

    /// Map backward from each point to the basic block that it
    /// belongs to.
    basic_blocks: IndexVec<PointIndex, BasicBlock>,

    num_points: usize,
}

impl DenseLocationMap {
    #[inline]
    pub fn new(body: &Body<'_>) -> Self {
        let mut num_points = 0;
        let statements_before_block: IndexVec<BasicBlock, usize> = body
            .basic_blocks
            .iter()
            .map(|block_data| {
                let v = num_points;
                num_points += block_data.statements.len() + 1;
                v
            })
            .collect();

        let mut basic_blocks = IndexVec::with_capacity(num_points);
        for (bb, bb_data) in body.basic_blocks.iter_enumerated() {
            basic_blocks.extend((0..=bb_data.statements.len()).map(|_| bb));
        }

        Self { statements_before_block, basic_blocks, num_points }
    }

    /// Total number of point indices
    #[inline]
    pub fn num_points(&self) -> usize {
        self.num_points
    }

    /// Converts a `Location` into a `PointIndex`. O(1).
    #[inline]
    pub fn point_from_location(&self, location: Location) -> PointIndex {
        let Location { block, statement_index } = location;
        let start_index = self.statements_before_block[block];
        PointIndex::new(start_index + statement_index)
    }

    /// Returns the `PointIndex` for the first statement in the given `BasicBlock`. O(1).
    #[inline]
    pub fn entry_point(&self, block: BasicBlock) -> PointIndex {
        let start_index = self.statements_before_block[block];
        PointIndex::new(start_index)
    }

    /// Return the PointIndex for the block start of this index.
    #[inline]
    pub fn to_block_start(&self, index: PointIndex) -> PointIndex {
        PointIndex::new(self.statements_before_block[self.basic_blocks[index]])
    }

    /// Converts a `PointIndex` back to a location. O(1).
    #[inline]
    pub fn to_location(&self, index: PointIndex) -> Location {
        assert!(index.index() < self.num_points);
        let block = self.basic_blocks[index];
        let start_index = self.statements_before_block[block];
        let statement_index = index.index() - start_index;
        Location { block, statement_index }
    }

    /// Sometimes we get point-indices back from bitsets that may be
    /// out of range (because they round up to the nearest 2^N number
    /// of bits). Use this function to filter such points out if you
    /// like.
    #[inline]
    pub fn point_in_range(&self, index: PointIndex) -> bool {
        index.index() < self.num_points
    }
}

rustc_index::newtype_index! {
    /// A single integer representing a `Location` in the MIR control-flow
    /// graph. Constructed efficiently from `DenseLocationMap`.
    #[orderable]
    #[debug_format = "PointIndex({})"]
    pub struct PointIndex {}
}
