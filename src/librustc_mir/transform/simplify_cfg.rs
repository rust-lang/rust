// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::middle::const_eval::ConstVal;
use rustc::mir::repr::*;
use std::mem;
use transform::util;
use transform::MirPass;

pub struct SimplifyCfg;

impl SimplifyCfg {
    pub fn new() -> SimplifyCfg {
        SimplifyCfg
    }

    fn merge_consecutive_blocks(&self, mir: &mut Mir) -> bool {

        // Find the target at the end of the jump chain, return None if there is a loop
        fn final_target(mir: &Mir, mut target: BasicBlock) -> Option<BasicBlock> {
            // Keep track of already seen blocks to detect loops
            let mut seen: Vec<BasicBlock> = Vec::with_capacity(8);

            while mir.basic_block_data(target).statements.is_empty() {
                match mir.basic_block_data(target).terminator {
                    Terminator::Goto { target: next } => {
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

        let mut predecessor_map = util::build_predecessor_map(mir);

        let mut changed = false;
        let mut seen = vec![false; mir.basic_blocks.len()];
        let mut worklist = vec![START_BLOCK];
        while let Some(bb) = worklist.pop() {
            // Temporarily swap out the terminator we're modifying to keep borrowck happy
            let mut terminator = Terminator::Diverge;
            mem::swap(&mut terminator, &mut mir.basic_block_data_mut(bb).terminator);

            // Shortcut chains of empty blocks that just jump from one to the next
            for target in terminator.successors_mut() {
                let new_target = match final_target(mir, *target) {
                    Some(new_target) => new_target,
                    None if mir.basic_block_data(bb).statements.is_empty() => bb,
                    None => continue
                };

                if *target != new_target {
                    changed = true;
                    predecessor_map.remove_predecessor(*target);
                    predecessor_map.add_predecessor(new_target);
                    *target = new_target;
                }
            }

            // See if we can merge the target block into this one
            match terminator {
                Terminator::Goto { target } if target.index() > DIVERGE_BLOCK.index() &&
                        predecessor_map.num_predecessors(target) == 1 => {
                    changed = true;
                    let mut other_data = BasicBlockData {
                        statements: Vec::new(),
                        terminator: Terminator::Goto { target: target}
                    };
                    mem::swap(&mut other_data, mir.basic_block_data_mut(target));

                    // target used to have 1 predecessor (bb), and still has only one (itself)
                    // All the successors of target have had target replaced by bb in their
                    // list of predecessors, keeping the number the same.

                    let data = mir.basic_block_data_mut(bb);
                    data.statements.append(&mut other_data.statements);
                    mem::swap(&mut data.terminator, &mut other_data.terminator);
                }
                _ => mir.basic_block_data_mut(bb).terminator = terminator
            }

            for succ in mir.basic_block_data(bb).terminator.successors() {
                if !seen[succ.index()] {
                    seen[succ.index()] = true;
                    worklist.push(*succ);
                }
            }
        }

        // These blocks must be retained, so mark them seen even if we didn't see them
        seen[START_BLOCK.index()] = true;
        seen[END_BLOCK.index()] = true;
        seen[DIVERGE_BLOCK.index()] = true;

        // Now get rid of all the blocks we never saw
        util::retain_basic_blocks(mir, &seen);

        changed
    }

    fn simplify_branches(&self, mir: &mut Mir) -> bool {
        let mut changed = false;

        for bb in mir.all_basic_blocks() {
            // Temporarily swap out the terminator we're modifying to keep borrowck happy
            let mut terminator = Terminator::Diverge;
            mem::swap(&mut terminator, &mut mir.basic_block_data_mut(bb).terminator);

            mir.basic_block_data_mut(bb).terminator = match terminator {
                Terminator::If { ref targets, .. } if targets[0] == targets[1] => {
                    changed = true;
                    Terminator::Goto { target: targets[0] }
                }
                Terminator::If { ref targets, cond: Operand::Constant(Constant {
                    literal: Literal::Value {
                        value: ConstVal::Bool(cond)
                    }, ..
                }) } => {
                    changed = true;
                    let target_idx = if cond { 0 } else { 1 };
                    Terminator::Goto { target: targets[target_idx] }
                }
                Terminator::SwitchInt { ref targets, .. }  if targets.len() == 1 => {
                    Terminator::Goto { target: targets[0] }
                }
                _ => terminator
            }
        }

        changed
    }
}

impl<'tcx> MirPass<'tcx> for SimplifyCfg {
    fn run_on_mir(&mut self, mir: &mut Mir<'tcx>) {
        let mut changed = true;
        while changed {
            changed = self.simplify_branches(mir);
            changed |= self.merge_consecutive_blocks(mir);
        }

        // FIXME: Should probably be moved into some kind of pass manager
        mir.basic_blocks.shrink_to_fit();
    }
}
