// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A pass that erases the contents of dead blocks. This pass must
//! run before any analysis passes because some of the dead blocks
//! can be ill-typed.
//!
//! The main problem is that typeck lets most blocks whose end is not
//! reachable have an arbitrary return type, rather than having the
//! usual () return type (as a note, typeck's notion of reachability
//! is in fact slightly weaker than MIR CFG reachability - see #31617).
//!
//! A standard example of the situation is:
//! ```rust
//!   fn example() {
//!       let _a: char = { return; };
//!   }
//! ```
//!
//! Here the block (`{ return; }`) has the return type `char`,
//! rather than `()`, but the MIR we naively generate still contains
//! the `_a = ()` write in the unreachable block "after" the return.
//!
//! As we have to run this pass even when we want to debug the MIR,
//! this pass just replaces the blocks with empty "return" blocks
//! and does not renumber anything.

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

        for (n, (block, seen)) in mir.basic_blocks.iter_mut().zip(seen).enumerate() {
            if !seen {
                info!("clearing block #{}: {:?}", n, block);
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
