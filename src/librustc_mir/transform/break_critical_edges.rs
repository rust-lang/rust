// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use std::mem;

use rustc_back::slice;
use rustc::mir::repr::*;
use rustc::mir::mir_map::MirMap;

use traversal;

/**
 * Breaks critical edges in the MIR.
 *
 * Critical edges are edges that are neither the only edge leaving a
 * block, nor the only edge entering one.
 *
 * When you want something to happen "along" an edge, you can either
 * do at the end of the predecessor block, or at the start of the
 * successor block. Critical edges have to be broken in order to prevent
 * "edge actions" from affecting other edges.
 *
 * This function will break those edges by inserting new blocks along them.
 *
 * A special case is Drop and Call terminators with unwind/cleanup successors,
 * They use `invoke` in LLVM, which terminates a block, meaning that code cannot
 * be inserted after them, so even if an edge is the only edge leaving a block
 * like that, we still insert blocks if the edge is one of many entering the
 * target.
 *
 * NOTE: Simplify CFG will happily undo most of the work this pass does.
 *
 */
pub fn break_critical_edges<'tcx>(mir_map: &mut MirMap<'tcx>) {
    for (_, mir) in &mut mir_map.map {
        break_critical_edges_fn(mir);
    }
}

/*
 * Predecessor map for tracking the predecessors of a block
 */
struct PredMap {
    preds: Vec<BlockPredecessors>
}

/**
 * Most blocks only have one predecessor, so we can cut down on
 * some allocation by not using Vec until we have more than one.
 */
#[derive(Clone)]
enum BlockPredecessors {
    None,
    One(BasicBlock),
    Some(Vec<BasicBlock>)
}

impl PredMap {
    pub fn new(n: usize) -> PredMap {
        let preds = vec![BlockPredecessors::None; n];

        PredMap {
            preds: preds
        }
    }

    fn ensure_len(&mut self, bb: BasicBlock) {
        let idx = bb.index();
        while self.preds.len() <= idx {
            self.preds.push(BlockPredecessors::None);
        }
    }

    pub fn add_pred(&mut self, target: BasicBlock, pred: BasicBlock) {
        self.ensure_len(target);

        let preds = mem::replace(&mut self.preds[target.index()], BlockPredecessors::None);
        match preds {
            BlockPredecessors::None => {
                self.preds[target.index()] = BlockPredecessors::One(pred);
            }
            BlockPredecessors::One(bb) => {
                self.preds[target.index()] = BlockPredecessors::Some(vec![bb, pred]);
            }
            BlockPredecessors::Some(mut preds) => {
                preds.push(pred);
                self.preds[target.index()] = BlockPredecessors::Some(preds);
            }
        }
    }

    pub fn remove_pred(&mut self, target: BasicBlock, pred: BasicBlock) {
        self.ensure_len(target);

        let preds = mem::replace(&mut self.preds[target.index()], BlockPredecessors::None);
        match preds {
            BlockPredecessors::None => {}
            BlockPredecessors::One(bb) if bb == pred => {}

            BlockPredecessors::One(bb) => {
                self.preds[target.index()] = BlockPredecessors::One(bb);
            }

            BlockPredecessors::Some(mut preds) => {
                preds.retain(|&bb| bb != pred);
                self.preds[target.index()] = BlockPredecessors::Some(preds);
            }
        }
    }

    pub fn get_preds(&self, bb: BasicBlock) -> &[BasicBlock] {
        match self.preds[bb.index()] {
            BlockPredecessors::None => &[],
            BlockPredecessors::One(ref bb) => slice::ref_slice(bb),
            BlockPredecessors::Some(ref bbs) => &bbs[..]
        }
    }
}


fn break_critical_edges_fn(mir: &mut Mir) {
    let mut pred_map = PredMap::new(mir.basic_blocks.len());

    // Build the precedecessor map for the MIR
    for (pred, data) in traversal::preorder(mir) {
        if let Some(ref term) = data.terminator {
            for &tgt in term.successors().iter() {
                pred_map.add_pred(tgt, pred);
            }
        }
    }

    // We need a place to store the new blocks generated
    let mut new_blocks = Vec::new();

    let bbs = mir.all_basic_blocks();
    let cur_len = mir.basic_blocks.len();

    for &bb in &bbs {
        let data = mir.basic_block_data_mut(bb);

        if let Some(ref mut term) = data.terminator {
            let is_invoke = term_is_invoke(term);
            let succs = term.successors_mut();
            if succs.len() > 1 || (succs.len() > 0 && is_invoke) {
                for tgt in succs {
                    let num_preds = pred_map.get_preds(*tgt).len();
                    if num_preds > 1 {
                        // It's a critical edge, break it
                        let goto = Terminator::Goto { target: *tgt };
                        let data = BasicBlockData::new(Some(goto));
                        // Get the index it will be when inserted into the MIR
                        let idx = cur_len + new_blocks.len();
                        new_blocks.push(data);
                        *tgt = BasicBlock::new(idx);
                    }
                }
            }
        }
    }

    debug!("Broke {} N edges", new_blocks.len());

    mir.basic_blocks.extend_from_slice(&new_blocks);
}

// Returns true if the terminator would use an invoke in LLVM.
fn term_is_invoke(term: &Terminator) -> bool {
    match *term {
        Terminator::Call { cleanup: Some(_), .. } |
        Terminator::Drop { unwind: Some(_), .. } => true,
        _ => false
    }
}
