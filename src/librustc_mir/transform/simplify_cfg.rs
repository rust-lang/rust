// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc_data_structures::bitvec::BitVector;
use rustc::middle::const_val::ConstVal;
use rustc::ty::TyCtxt;
use rustc::mir::repr::*;
use rustc::mir::transform::{MirPass, MirSource, Pass};
use pretty;
use std::mem;

use super::predecessor_map::PredecessorMap;
use super::remove_dead_blocks::RemoveDeadBlocks;

pub struct SimplifyCfg;

impl SimplifyCfg {
    pub fn new() -> SimplifyCfg {
        SimplifyCfg
    }
}

impl<'tcx> MirPass<'tcx> for SimplifyCfg {
    fn run_pass<'a>(&mut self, tcx: TyCtxt<'a, 'tcx, 'tcx>, src: MirSource, mir: &mut Mir<'tcx>) {
        simplify_branches(mir);
        RemoveDeadBlocks.run_pass(tcx, src, mir);
        merge_consecutive_blocks(mir);
        RemoveDeadBlocks.run_pass(tcx, src, mir);
        pretty::dump_mir(tcx, "simplify_cfg", &0, src, mir, None);

        // FIXME: Should probably be moved into some kind of pass manager
        mir.basic_blocks.shrink_to_fit();
    }
}

impl Pass for SimplifyCfg {}

fn merge_consecutive_blocks(mir: &mut Mir) {
    let mut predecessor_map = PredecessorMap::from_mir(mir);

    loop {
        let mut changed = false;
        let mut seen = BitVector::new(mir.basic_blocks.len());
        let mut worklist = vec![START_BLOCK];
        while let Some(bb) = worklist.pop() {
            // Temporarily take ownership of the terminator we're modifying to keep borrowck happy
            let mut terminator = mir.basic_block_data_mut(bb).terminator.take()
                .expect("invalid terminator state");

            // See if we can merge the target block into this one
            loop {
                let mut inner_change = false;

                if let TerminatorKind::Goto { target } = terminator.kind {
                    // Don't bother trying to merge a block into itself
                    if target == bb {
                        break;
                    }

                    let num_preds = predecessor_map.predecessors(target).len();
                    let num_insts = mir.basic_block_data(target).statements.len();
                    match mir.basic_block_data(target).terminator().kind {
                        _ if num_preds == 1 => {
                            inner_change = true;
                            let mut stmts = Vec::new();
                            {
                                let target_data = mir.basic_block_data_mut(target);
                                mem::swap(&mut stmts, &mut target_data.statements);
                                mem::swap(&mut terminator, target_data.terminator_mut());
                            }

                            mir.basic_block_data_mut(bb).statements.append(&mut stmts);

                            predecessor_map.replace_predecessor(target, bb, target);
                            for succ in terminator.successors().iter() {
                                predecessor_map.replace_predecessor(*succ, target, bb);
                            }
                        }
                        TerminatorKind::Goto { target: new_target } if num_insts == 0 => {
                            inner_change = true;
                            terminator.kind = TerminatorKind::Goto { target: new_target };
                            predecessor_map.replace_successor(bb, target, new_target);
                        }
                        _ => {}
                    };
                }

                for target in terminator.successors_mut() {
                    let new_target = match final_target(mir, *target) {
                        Some(new_target) => new_target,
                        None if mir.basic_block_data(bb).statements.is_empty() => bb,
                        None => continue
                    };
                    if *target != new_target {
                        inner_change = true;
                        predecessor_map.replace_successor(bb, *target, new_target);
                        *target = new_target;
                    }
                }

                changed |= inner_change;
                if !inner_change {
                    break;
                }
            }

            mir.basic_block_data_mut(bb).terminator = Some(terminator);

            for succ in mir.basic_block_data(bb).terminator().successors().iter() {
                if seen.insert(succ.index()) {
                    worklist.push(*succ);
                }
            }
        }

        if !changed {
            break;
        }
    }
}

// Find the target at the end of the jump chain, return None if there is a loop
fn final_target(mir: &Mir, mut target: BasicBlock) -> Option<BasicBlock> {
    // Keep track of already seen blocks to detect loops
    let mut seen: Vec<BasicBlock> = Vec::with_capacity(8);

    while mir.basic_block_data(target).statements.is_empty() {
        // NB -- terminator may have been swapped with `None` in
        // merge_consecutive_blocks, in which case we have a cycle and just want
        // to stop
        match mir.basic_block_data(target).terminator {
            Some(Terminator { kind: TerminatorKind::Goto { target: next }, .. }) =>  {
                if seen.contains(&next) {
                    return None;
                }
                seen.push(next);
                target = next;
            }
            _ => break
        }
    }

    Some(target)
}

fn simplify_branches(mir: &mut Mir) {
    loop {
        let mut changed = false;

        for bb in mir.all_basic_blocks() {
            let basic_block = mir.basic_block_data_mut(bb);
            let mut terminator = basic_block.terminator_mut();
            terminator.kind = match terminator.kind {
                TerminatorKind::If { ref targets, .. } if targets.0 == targets.1 => {
                    changed = true;
                    TerminatorKind::Goto { target: targets.0 }
                }

                TerminatorKind::If { ref targets, cond: Operand::Constant(Constant {
                    literal: Literal::Value {
                        value: ConstVal::Bool(cond)
                    }, ..
                }) } => {
                    changed = true;
                    if cond {
                        TerminatorKind::Goto { target: targets.0 }
                    } else {
                        TerminatorKind::Goto { target: targets.1 }
                    }
                }

                TerminatorKind::SwitchInt { ref targets, .. } if targets.len() == 1 => {
                    TerminatorKind::Goto { target: targets[0] }
                }
                _ => continue
            }
        }

        if !changed {
            break;
        }
    }
}
