// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::middle::const_val::ConstVal;
use rustc::ty::TyCtxt;
use rustc::mir::repr::*;
use rustc::mir::transform::{MirPass, Pass};
use pretty;
use syntax::ast::NodeId;

use super::remove_dead_blocks::RemoveDeadBlocks;

pub struct SimplifyCfg;

impl SimplifyCfg {
    pub fn new() -> SimplifyCfg {
        SimplifyCfg
    }

    fn remove_goto_chains(&self, mir: &mut Mir) -> bool {
        // Find the target at the end of the jump chain, return None if there is a loop
        fn final_target(mir: &Mir, mut target: BasicBlock) -> Option<BasicBlock> {
            // Keep track of already seen blocks to detect loops
            let mut seen: Vec<BasicBlock> = Vec::with_capacity(8);

            while mir.basic_block_data(target).statements.is_empty() {
                // NB -- terminator may have been swapped with `None`
                // below, in which case we have a cycle and just want
                // to stop
                if let Some(ref terminator) = mir.basic_block_data(target).terminator {
                    match terminator.kind {
                        TerminatorKind::Goto { target: next } => {
                            if seen.contains(&next) {
                                return None;
                            }
                            seen.push(next);
                            target = next;
                        }
                        _ => break
                    }
                } else {
                    break
                }
            }

            Some(target)
        }

        let mut changed = false;
        for bb in mir.all_basic_blocks() {
            // Temporarily take ownership of the terminator we're modifying to keep borrowck happy
            let mut terminator = mir.basic_block_data_mut(bb).terminator.take()
                                    .expect("invalid terminator state");

            debug!("remove_goto_chains: bb={:?} terminator={:?}", bb, terminator);

            for target in terminator.successors_mut() {
                let new_target = match final_target(mir, *target) {
                    Some(new_target) => new_target,
                    None if mir.basic_block_data(bb).statements.is_empty() => bb,
                    None => continue
                };
                changed |= *target != new_target;
                *target = new_target;
            }
            mir.basic_block_data_mut(bb).terminator = Some(terminator);
        }
        changed
    }

    fn simplify_branches(&self, mir: &mut Mir) -> bool {
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

        changed
    }
}

impl<'tcx> MirPass<'tcx> for SimplifyCfg {
    fn run_pass(&mut self, tcx: &TyCtxt<'tcx>, id: NodeId, mir: &mut Mir<'tcx>) {
        let mut counter = 0;
        let mut changed = true;
        while changed {
            pretty::dump_mir(tcx, "simplify_cfg", &counter, id, mir, None);
            counter += 1;
            changed = self.simplify_branches(mir);
            changed |= self.remove_goto_chains(mir);
            RemoveDeadBlocks.run_pass(tcx, id, mir);
        }
        // FIXME: Should probably be moved into some kind of pass manager
        mir.basic_blocks.shrink_to_fit();
    }
}

impl Pass for SimplifyCfg {}
