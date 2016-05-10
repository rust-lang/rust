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
use rustc::mir::transform::{MirPass, MirSource, Pass};

use rustc_data_structures::bitvec::BitVector;

use pretty;

use traversal;

pub struct BreakCleanupEdges;

/**
 * Breaks outgoing critical edges for call terminators in the MIR.
 *
 * Critical edges are edges that are neither the only edge leaving a
 * block, nor the only edge entering one.
 *
 * When you want something to happen "along" an edge, you can either
 * do at the end of the predecessor block, or at the start of the
 * successor block. Critical edges have to be broken in order to prevent
 * "edge actions" from affecting other edges. We need this for calls that are
 * translated to LLVM invoke instructions, because invoke is a block terminator
 * in LLVM so we can't insert any code to handle the call's result into the
 * block that performs the call.
 *
 * This function will break those edges by inserting new blocks along them.
 *
 * NOTE: Simplify CFG will happily undo most of the work this pass does.
 *
 */

impl<'tcx> MirPass<'tcx> for BreakCleanupEdges {
    fn run_pass<'a>(&mut self, tcx: TyCtxt<'a, 'tcx, 'tcx>, src: MirSource, mir: &mut Mir<'tcx>) {
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
                if term_is_invoke(term) {
                    let term_span = term.span;
                    let term_scope = term.scope;
                    let succs = term.successors_mut();
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

        pretty::dump_mir(tcx, "break_cleanup_edges", &0, src, mir, None);
        debug!("Broke {} N edges", new_blocks.len());

        mir.basic_blocks.extend_from_slice(&new_blocks);
    }
}

impl Pass for BreakCleanupEdges {}

// Returns true if the terminator is a call that would use an invoke in LLVM.
fn term_is_invoke(term: &Terminator) -> bool {
    match term.kind {
        TerminatorKind::Call { cleanup: Some(_), .. } |
        TerminatorKind::Drop { unwind: Some(_), .. } => true,
        _ => false
    }
}
