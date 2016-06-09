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
use rustc::mir::traversal;
use pretty;

pub struct AddCallGuards;

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

impl<'tcx> MirPass<'tcx> for AddCallGuards {
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

        // We need a place to store the new blocks generated
        let mut new_blocks = Vec::new();

        let bbs = mir.all_basic_blocks();
        let cur_len = mir.basic_blocks.len();

        for &bb in &bbs {
            let data = mir.basic_block_data_mut(bb);

            match data.terminator {
                Some(Terminator {
                    kind: TerminatorKind::Call {
                        destination: Some((_, ref mut destination)),
                        cleanup: Some(_),
                        ..
                    }, source_info
                }) if pred_count[destination.index()] > 1 => {
                    // It's a critical edge, break it
                    let call_guard = BasicBlockData {
                        statements: vec![],
                        is_cleanup: data.is_cleanup,
                        terminator: Some(Terminator {
                            source_info: source_info,
                            kind: TerminatorKind::Goto { target: *destination }
                        })
                    };

                    // Get the index it will be when inserted into the MIR
                    let idx = cur_len + new_blocks.len();
                    new_blocks.push(call_guard);
                    *destination = BasicBlock::new(idx);
                }
                _ => {}
            }
        }

        pretty::dump_mir(tcx, "break_cleanup_edges", &0, src, mir, None);
        debug!("Broke {} N edges", new_blocks.len());

        mir.basic_blocks.extend_from_slice(&new_blocks);
    }
}

impl Pass for AddCallGuards {}
