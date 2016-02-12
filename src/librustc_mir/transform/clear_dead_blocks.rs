// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A pass that erases the contents of dead blocks. This is required
//! because rustc allows for ill-typed block terminators in dead
//! blocks.
//!
//! This pass does not renumber or remove the blocks, to have the
//! MIR better match the source.

use rustc::middle::infer;
use rustc::mir::repr::*;
use rustc::mir::transform::MirPass;

pub struct ClearDeadBlocks;

impl ClearDeadBlocks {
    pub fn new() -> ClearDeadBlocks {
        ClearDeadBlocks
    }

    fn clear_dead_blocks(&self, mir: &mut Mir) {
        let mut seen = vec![false; mir.basic_blocks.len()];

        // These blocks are always required.
        seen[START_BLOCK.index()] = true;
        seen[END_BLOCK.index()] = true;

        let mut worklist = vec![START_BLOCK];
        while let Some(bb) = worklist.pop() {
            for succ in mir.basic_block_data(bb).terminator().successors().iter() {
                if !seen[succ.index()] {
                    seen[succ.index()] = true;
                    worklist.push(*succ);
                }
            }
        }

        for (block, seen) in mir.basic_blocks.iter_mut().zip(seen) {
            if !seen {
                *block = BasicBlockData {
                    statements: vec![],
                    terminator: Some(Terminator::Return),
                    is_cleanup: false
                };
            }
        }
    }
}

impl MirPass for ClearDeadBlocks {
    fn run_on_mir<'a, 'tcx>(&mut self, mir: &mut Mir<'tcx>, _: &infer::InferCtxt<'a, 'tcx>)
    {
        self.clear_dead_blocks(mir);
    }
}
