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
use rustc::mir::*;
use rustc::mir::transform::{MirPass, MirSource, Pass};
use rustc_data_structures::indexed_vec::Idx;

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
    fn run_pass<'a>(&mut self, _tcx: TyCtxt<'a, 'tcx, 'tcx>, _src: MirSource, mir: &mut Mir<'tcx>) {
        add_call_guards(mir);
    }
}

pub fn add_call_guards(mir: &mut Mir) {
    // We need a place to store the new blocks generated
    let mut new_blocks = Vec::new();

    let cur_len = mir.basic_blocks().len();

    for block in mir.basic_blocks_mut() {
        // Call statement indices, since the last call.
        let mut calls = Vec::new();
        // Iterate in reverse to allow draining from the end of statements, not the middle
        for i in (0..block.statements.len()).rev() {
            if let StatementKind::Call { .. } = block.statements[i].kind {
                calls.push(i);
            }
        }

        let first_new_block_idx = cur_len + new_blocks.len();
        let mut new_blocks_iter = Vec::new();

        debug!("original statements = {:#?}", block.statements);

        let mut is_first = true;

        for &el in calls.iter() {
            let after_call = block.statements.split_off(el + 1);

            let next_block_idx = first_new_block_idx + new_blocks_iter.len();
            let terminator = if is_first {
                block.terminator.take().expect("invalid terminator state")
            } else {
                Terminator {
                    source_info: after_call[0].source_info,
                    kind: TerminatorKind::Goto { target: Block::new(next_block_idx - 1) }
                }
            };

            debug!("cg: statements = {:?}", after_call);
            let call_guard = BlockData {
                statements: after_call,
                is_cleanup: block.is_cleanup,
                terminator: Some(terminator)
            };

            new_blocks_iter.push(call_guard);
            is_first = false;
        }

        debug!("after blocks = {:#?}", new_blocks_iter);

        for bb_data in &new_blocks_iter {
            let c = bb_data.statements.iter().filter(|stmt| {
                match stmt.kind {
                    StatementKind::Call { .. } => true,
                    _ => false,
                }
            }).count();
            assert!(c <= 1, "{} calls in {:?}", c, bb_data);
        }

        if !new_blocks_iter.is_empty() {
            block.terminator = Some(Terminator {
                source_info: new_blocks_iter[0].terminator().source_info,
                kind: TerminatorKind::Goto {
                    target: Block::new(first_new_block_idx + new_blocks_iter.len() - 1)
                }
            });
        }

        new_blocks.extend(new_blocks_iter);
    }

    debug!("Broke {} N edges", new_blocks.len());

    mir.basic_blocks_mut().extend(new_blocks);
}

impl Pass for AddCallGuards {}
