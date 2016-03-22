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

use rustc_data_structures::bitvec::BitVector;
use rustc::ty::TyCtxt;
use rustc::mir::repr::*;
use rustc::mir::transform::{Pass, MirPass};
use syntax::ast::NodeId;

pub struct RemoveDeadBlocks;

impl<'tcx> MirPass<'tcx> for RemoveDeadBlocks {
    fn run_pass(&mut self, _: &TyCtxt<'tcx>, _: NodeId, mir: &mut Mir<'tcx>) {
        let mut seen = BitVector::new(mir.basic_blocks.len());
        // These blocks are always required.
        seen.insert(START_BLOCK.index());
        seen.insert(END_BLOCK.index());

        let mut worklist = Vec::with_capacity(4);
        worklist.push(START_BLOCK);
        while let Some(bb) = worklist.pop() {
            for succ in mir.basic_block_data(bb).terminator().successors().iter() {
                if seen.insert(succ.index()) {
                    worklist.push(*succ);
                }
            }
        }
        retain_basic_blocks(mir, &seen);
    }
}

impl Pass for RemoveDeadBlocks {}

/// Mass removal of basic blocks to keep the ID-remapping cheap.
fn retain_basic_blocks(mir: &mut Mir, keep: &BitVector) {
    let num_blocks = mir.basic_blocks.len();

    let mut replacements: Vec<_> = (0..num_blocks).map(BasicBlock::new).collect();
    let mut used_blocks = 0;
    for alive_index in keep.iter() {
        replacements[alive_index] = BasicBlock::new(used_blocks);
        if alive_index != used_blocks {
            // Swap the next alive block data with the current available slot. Since alive_index is
            // non-decreasing this is a valid operation.
            mir.basic_blocks.swap(alive_index, used_blocks);
        }
        used_blocks += 1;
    }
    mir.basic_blocks.truncate(used_blocks);

    for bb in mir.all_basic_blocks() {
        for target in mir.basic_block_data_mut(bb).terminator_mut().successors_mut() {
            *target = replacements[target.index()];
        }
    }
}
