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
use rustc::middle::ty;
use rustc::mir::repr::*;
use rustc::mir::visit::MutVisitor;
use transform::util;
use rustc::mir::transform::{MirPass, Pass};

pub struct SimplifyCfg;

impl Pass for SimplifyCfg {
    fn priority(&self) -> usize {
        50
    }
}

impl MirPass for SimplifyCfg {
    fn run_pass<'tcx>(&mut self, tcx: &ty::ctxt<'tcx>, mir: &mut Mir<'tcx>) {
        let mut dbr_pass = DeadBlockRemoval::new(self);
        let mut changed = true;
        while changed {
            changed = self.simplify_branches(mir);
            changed |= self.remove_goto_chains(mir);
            dbr_pass.run_pass(tcx, mir);
        }
    }
}

impl SimplifyCfg {
    fn remove_goto_chains(&self, mir: &mut Mir) -> bool {

        // Find the target at the end of the jump chain, return None if there is a loop
        fn final_target(mir: &Mir, mut target: BasicBlock) -> Option<BasicBlock> {
            // Keep track of already seen blocks to detect loops
            let mut seen: Vec<BasicBlock> = Vec::with_capacity(8);

            while mir.basic_block_data(target).statements.is_empty() {
                match mir.basic_block_data(target).terminator {
                    Some(Terminator::Goto { target: next }) => {
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

        let mut changed = false;
        for bb in mir.all_basic_blocks() {
            // Temporarily take ownership of the terminator we're modifying to keep borrowck happy
            let mut terminator = mir.basic_block_data_mut(bb).terminator.take()
                                    .expect("invalid terminator state");

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

    // FIXME: This transformation should be interleaved with the constant-propagation pass.
    fn simplify_branches(&self, mir: &mut Mir) -> bool {
        let mut changed = false;

        for bb in mir.all_basic_blocks() {
            let basic_block = mir.basic_block_data_mut(bb);
            let mut terminator = basic_block.terminator_mut();

            *terminator = match *terminator {
                Terminator::If { ref targets, .. } if targets.0 == targets.1 => {
                    changed = true;
                    Terminator::Goto { target: targets.0 }
                }
                Terminator::If { ref targets, cond: Operand::Constant(Constant {
                    literal: Literal::Value {
                        value: ConstVal::Bool(cond)
                    }, ..
                }) } => {
                    changed = true;
                    if cond {
                        Terminator::Goto { target: targets.0 }
                    } else {
                        Terminator::Goto { target: targets.1 }
                    }
                }
                Terminator::SwitchInt { ref targets, .. }  if targets.len() == 1 => {
                    Terminator::Goto { target: targets[0] }
                }
                _ => continue
            }
        }

        changed
    }
}

/// Remove all the unreachable blocks.
///
/// You want to schedule this pass just after any pass which might introduce unreachable blocks.
/// This pass is very cheap and might improve the run-time of other not-so-cheap passes.
pub struct DeadBlockRemoval(usize);

impl DeadBlockRemoval {
    fn new(run_after: &Pass) -> DeadBlockRemoval {
        DeadBlockRemoval(run_after.priority() + 1)
    }
}

impl Pass for DeadBlockRemoval {
    fn priority(&self) -> usize {
        self.0
    }
}

impl MirPass for DeadBlockRemoval {
    fn run_pass<'tcx>(&mut self, _: &ty::ctxt<'tcx>, mir: &mut Mir<'tcx>) {
        let mut seen = vec![false; mir.basic_blocks.len()];

        // These blocks are always required.
        seen[START_BLOCK.index()] = true;
        seen[END_BLOCK.index()] = true;

        let mut worklist = vec![START_BLOCK];
        while let Some(bb) = worklist.pop() {
            for succ in mir.basic_block_data(bb).terminator().successors().iter() {
                if !seen[succ.index()] {
                    seen[succ.index()] = true;
                    worklist.push(*succ);
                }
            }
        }

        util::retain_basic_blocks(mir, &seen);
    }
}

/// Reduce the memory allocated by a MIR graph.
pub struct CompactMir;

impl Pass for CompactMir {
    fn priority(&self) -> usize {
        // We want this pass to run very late, so we give it a very high priority number.
        !10
    }
}

impl MirPass for CompactMir {
    fn run_pass<'tcx>(&mut self, _: &ty::ctxt<'tcx>, mir: &mut Mir<'tcx>) {
        self.visit_mir(mir);
    }
}

impl<'tcx> MutVisitor<'tcx> for CompactMir {
    fn visit_mir(&mut self, mir: &mut Mir<'tcx>) {
        mir.basic_blocks.shrink_to_fit();
        mir.var_decls.shrink_to_fit();
        mir.arg_decls.shrink_to_fit();
        mir.temp_decls.shrink_to_fit();
        self.super_mir(mir);
    }

    fn visit_basic_block_data(&mut self, b: BasicBlock, data: &mut BasicBlockData) {
        data.statements.shrink_to_fit();
        self.super_basic_block_data(b, data);
    }

    fn visit_terminator(&mut self, b: BasicBlock, terminator: &mut Terminator) {
        match *terminator {
            Terminator::Switch { ref mut targets, .. } => targets.shrink_to_fit(),
            Terminator::SwitchInt { ref mut values, ref mut targets, .. } => {
                values.shrink_to_fit();
                targets.shrink_to_fit();
            },
            Terminator::Call { ref mut args, .. } => args.shrink_to_fit(),
            _ => {/* nothing to do */}
        }
        self.super_terminator(b, terminator);
    }

    fn visit_rvalue(&mut self, rvalue: &mut Rvalue) {
        match *rvalue {
            Rvalue::Aggregate(_, ref mut operands) => operands.shrink_to_fit(),
            _ => { /* nothing to do */ }
        }
        self.super_rvalue(rvalue);
    }
}
