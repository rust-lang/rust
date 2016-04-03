// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::ty::TyCtxt;
use rustc::mir::repr::*;
use rustc::mir::transform::{MirPass, Pass};
use syntax::ast::NodeId;

use rustc_data_structures::bitvec::BitVector;

use traversal;

pub struct BreakCriticalEdges;

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

impl<'tcx> MirPass<'tcx> for BreakCriticalEdges {
    fn run_pass(&mut self, _: &TyCtxt<'tcx>, _: NodeId, mir: &mut Mir<'tcx>) {
        break_critical_edges(mir);
    }
}

impl Pass for BreakCriticalEdges {}

fn break_critical_edges(mir: &mut Mir) {
    let mut pred_count = vec![0u32; mir.basic_blocks.len()];

    // Build the precedecessor map for the MIR
    for (_, data) in traversal::preorder(mir) {
        if let Some(ref term) = data.terminator {
            for &tgt in term.successors().iter() {
                pred_count[tgt.index()] += 1;
            }
        }
    }

    let cleanup_map : BitVector = mir.basic_blocks
        .iter().map(|bb| bb.is_cleanup).collect();

    // We need a place to store the new blocks generated
    let mut new_blocks = Vec::new();

    let bbs = mir.all_basic_blocks();
    let cur_len = mir.basic_blocks.len();

    for &bb in &bbs {
        let data = mir.basic_block_data_mut(bb);

        if let Some(ref mut term) = data.terminator {
            let is_invoke = term_is_invoke(term);
            let term_span = term.span;
            let term_scope = term.scope;
            let succs = term.successors_mut();
            if succs.len() > 1 || (succs.len() > 0 && is_invoke) {
                for tgt in succs {
                    let num_preds = pred_count[tgt.index()];
                    if num_preds > 1 {
                        // It's a critical edge, break it
                        let goto = Terminator {
                            span: term_span,
                            scope: term_scope,
                            kind: TerminatorKind::Goto { target: *tgt }
                        };
                        let mut data = BasicBlockData::new(Some(goto));
                        data.is_cleanup = cleanup_map.contains(tgt.index());

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
    match term.kind {
        TerminatorKind::Call { cleanup: Some(_), .. } |
        TerminatorKind::Drop { unwind: Some(_), .. } => true,
        _ => false
    }
}
