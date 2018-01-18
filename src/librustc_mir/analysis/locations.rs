// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc_data_structures::indexed_vec::{IndexVec, Idx};
use rustc::mir::*;
use std::ops::Range;

newtype_index!(FlatLocation { DEBUG_FORMAT = "FlatLocation({})" });

/// Maps `Location`s containing a block index and a statement/terminator
/// index within the block, to a single linearized `FlatLocation` index.
pub struct FlatLocations {
    pub block_start: IndexVec<BasicBlock, FlatLocation>,
    pub total_count: usize
}

impl FlatLocations {
    pub fn collect(mir: &Mir) -> Self {
        let mut next_start = FlatLocation::new(0);
        FlatLocations {
            block_start: mir.basic_blocks().iter().map(|block| {
                let start = next_start;
                next_start = FlatLocation::new(start.index() + block.statements.len() + 1);
                start
            }).collect(),
            total_count: next_start.index()
        }
    }

    pub fn get(&self, location: Location) -> FlatLocation {
        let block_range = self.block_range(location.block);
        let id = FlatLocation::new(block_range.start.index() + location.statement_index);
        assert!(id < block_range.end);
        id
    }

    pub fn block_range(&self, block: BasicBlock) -> Range<FlatLocation> {
        let next_block = BasicBlock::new(block.index() + 1);
        let next_start = self.block_start.get(next_block).cloned()
            .unwrap_or(FlatLocation::new(self.total_count));
        self.block_start[block]..next_start
    }
}
