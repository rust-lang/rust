// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use repr::*;
use visit::*;

/// Update basic block ids in all terminators using the given replacements,
/// useful e.g. after removal of several basic blocks to update all terminators
/// in a single pass
pub fn update_basic_block_ids(mir: &mut Mir, replacements: &[BasicBlock]) {
    for bb in mir.all_basic_blocks() {
        for target in mir.basic_block_data_mut(bb).terminator.successors_mut() {
            *target = replacements[target.index()];
        }
    }
}

/// Mass removal of basic blocks to keep the ID-remapping cheap.
pub fn retain_basic_blocks(mir: &mut Mir, keep: &[bool]) {
    let num_blocks = mir.basic_blocks.len();

    // Check that we have a usage flag for every block
    assert_eq!(num_blocks, keep.len());

    let first_dead = match keep.iter().position(|&k| !k) {
        None => return,
        Some(first_dead) => first_dead,
    };

    // `replacements` maps the old block ids to the new ones
    let mut replacements: Vec<_> = (0..num_blocks).map(BasicBlock::new).collect();

    let mut dead = 0;
    for i in first_dead..num_blocks {
        if keep[i] {
            replacements[i] = BasicBlock::new(i - dead);
            mir.basic_blocks.swap(i, i - dead);
        } else {
            dead += 1;
        }
    }
    mir.basic_blocks.truncate(num_blocks - dead);

    update_basic_block_ids(mir, &replacements);
}

// A simple map to perform quick lookups of the predecessors of a BasicBlock.
// Since BasicBlocks usually only have a small number of predecessors, we use a
// simple vector. Also, if a block has the same target more than once, for
// example in a switch, it will appear in the target's predecessor list multiple
// times. This allows to update the map more easily when modifying the graph.
pub struct PredecessorMap {
    map: Vec<Vec<BasicBlock>>,
}

impl PredecessorMap {
    pub fn new(num_blocks: usize) -> PredecessorMap {
        PredecessorMap {
            map: vec![Vec::new(); num_blocks],
        }
    }

    pub fn predecessors(&self, block: BasicBlock) -> &[BasicBlock] {
        &self.map[block.index()]
    }

    pub fn add_predecessor(&mut self, block: BasicBlock, predecessor: BasicBlock) {
        self.map[block.index()].push(predecessor);
    }

    pub fn remove_predecessor(&mut self, block: BasicBlock, predecessor: BasicBlock) {
        let pos = self.map[block.index()].iter().position(|&p| p == predecessor).expect(
            &format!("{:?} is not registered as a predecessor of {:?}", predecessor, block));

        self.map[block.index()].swap_remove(pos);
    }

    pub fn replace_predecessor(&mut self, block: BasicBlock, old: BasicBlock, new: BasicBlock) {
        self.remove_predecessor(block, old);
        self.add_predecessor(block, new);
    }
}

struct PredecessorVisitor {
    predecessor_map: PredecessorMap,
}

impl PredecessorVisitor {
    fn new(num_blocks: usize) -> PredecessorVisitor {
        PredecessorVisitor {
            predecessor_map: PredecessorMap::new(num_blocks),
        }
    }
}

impl<'tcx> Visitor<'tcx> for PredecessorVisitor {
    fn visit_mir(&mut self, mir: &Mir<'tcx>) {
        self.predecessor_map = PredecessorMap::new(mir.basic_blocks.len());
        self.super_mir(mir);
    }

    fn visit_branch(&mut self, source: BasicBlock, target: BasicBlock) {
        self.predecessor_map.add_predecessor(target, source);
    }
}

pub fn build_predecessor_map(mir: &Mir) -> PredecessorMap {
    let mut v = PredecessorVisitor::new(mir.basic_blocks.len());
    v.visit_mir(mir);

    v.predecessor_map
}
