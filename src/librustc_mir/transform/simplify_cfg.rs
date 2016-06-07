// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A pass that removes various redundancies in the CFG. It should be
//! called after every significant CFG modification to tidy things
//! up.
//!
//! This pass must also be run before any analysis passes because it removes
//! dead blocks, and some of these can be ill-typed.
//!
//! The cause of that is that typeck lets most blocks whose end is not
//! reachable have an arbitrary return type, rather than having the
//! usual () return type (as a note, typeck's notion of reachability
//! is in fact slightly weaker than MIR CFG reachability - see #31617).
//!
//! A standard example of the situation is:
//! ```rust
//!   fn example() {
//!       let _a: char = { return; };
//!   }
//! ```
//!
//! Here the block (`{ return; }`) has the return type `char`,
//! rather than `()`, but the MIR we naively generate still contains
//! the `_a = ()` write in the unreachable block "after" the return.


use rustc_data_structures::bitvec::BitVector;
use rustc_data_structures::indexed_vec::{Idx, IndexVec};
use rustc::middle::const_val::ConstVal;
use rustc::ty::TyCtxt;
use rustc::mir::repr::*;
use rustc::mir::transform::{MirPass, MirSource, Pass};
use rustc::mir::traversal;
use pretty;
use std::mem;

pub struct SimplifyCfg<'a> { label: &'a str }

impl<'a> SimplifyCfg<'a> {
    pub fn new(label: &'a str) -> Self {
        SimplifyCfg { label: label }
    }
}

impl<'l, 'tcx> MirPass<'tcx> for SimplifyCfg<'l> {
    fn run_pass<'a>(&mut self, tcx: TyCtxt<'a, 'tcx, 'tcx>, src: MirSource, mir: &mut Mir<'tcx>) {
        pretty::dump_mir(tcx, "simplify_cfg", &format!("{}-before", self.label), src, mir, None);
        simplify_branches(mir);
        remove_dead_blocks(mir);
        merge_consecutive_blocks(mir);
        remove_dead_blocks(mir);
        pretty::dump_mir(tcx, "simplify_cfg", &format!("{}-after", self.label), src, mir, None);

        // FIXME: Should probably be moved into some kind of pass manager
        mir.basic_blocks_mut().raw.shrink_to_fit();
    }
}

impl<'l> Pass for SimplifyCfg<'l> {}

fn merge_consecutive_blocks(mir: &mut Mir) {
    // Build the precedecessor map for the MIR
    let mut pred_count = IndexVec::from_elem(0u32, mir.basic_blocks());
    for (_, data) in traversal::preorder(mir) {
        if let Some(ref term) = data.terminator {
            for &tgt in term.successors().iter() {
                pred_count[tgt] += 1;
            }
        }
    }

    loop {
        let mut changed = false;
        let mut seen = BitVector::new(mir.basic_blocks().len());
        let mut worklist = vec![START_BLOCK];
        while let Some(bb) = worklist.pop() {
            // Temporarily take ownership of the terminator we're modifying to keep borrowck happy
            let mut terminator = mir[bb].terminator.take().expect("invalid terminator state");

            // See if we can merge the target block into this one
            loop {
                let mut inner_change = false;

                if let TerminatorKind::Goto { target } = terminator.kind {
                    // Don't bother trying to merge a block into itself
                    if target == bb {
                        break;
                    }

                    let num_insts = mir[target].statements.len();
                    match mir[target].terminator().kind {
                        TerminatorKind::Goto { target: new_target } if num_insts == 0 => {
                            inner_change = true;
                            terminator.kind = TerminatorKind::Goto { target: new_target };
                            pred_count[target] -= 1;
                            pred_count[new_target] += 1;
                        }
                        _ if pred_count[target] == 1 => {
                            inner_change = true;
                            let mut stmts = Vec::new();
                            {
                                let target_data = &mut mir[target];
                                mem::swap(&mut stmts, &mut target_data.statements);
                                mem::swap(&mut terminator, target_data.terminator_mut());
                            }

                            mir[bb].statements.append(&mut stmts);
                        }
                        _ => {}
                    };
                }

                for target in terminator.successors_mut() {
                    let new_target = match final_target(mir, *target) {
                        Some(new_target) => new_target,
                        None if mir[bb].statements.is_empty() => bb,
                        None => continue
                    };
                    if *target != new_target {
                        inner_change = true;
                        pred_count[*target] -= 1;
                        pred_count[new_target] += 1;
                        *target = new_target;
                    }
                }

                changed |= inner_change;
                if !inner_change {
                    break;
                }
            }

            mir[bb].terminator = Some(terminator);

            for succ in mir[bb].terminator().successors().iter() {
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

    while mir[target].statements.is_empty() {
        // NB -- terminator may have been swapped with `None` in
        // merge_consecutive_blocks, in which case we have a cycle and just want
        // to stop
        match mir[target].terminator {
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

        for (_, basic_block) in mir.basic_blocks_mut().iter_enumerated_mut() {
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

                TerminatorKind::Assert { target, cond: Operand::Constant(Constant {
                    literal: Literal::Value {
                        value: ConstVal::Bool(cond)
                    }, ..
                }), expected, .. } if cond == expected => {
                    changed = true;
                    TerminatorKind::Goto { target: target }
                }

                TerminatorKind::SwitchInt { ref targets, .. } if targets.len() == 1 => {
                    changed = true;
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

fn remove_dead_blocks(mir: &mut Mir) {
    let mut seen = BitVector::new(mir.basic_blocks().len());
    for (bb, _) in traversal::preorder(mir) {
        seen.insert(bb.index());
    }

    let basic_blocks = mir.basic_blocks_mut();

    let num_blocks = basic_blocks.len();
    let mut replacements : Vec<_> = (0..num_blocks).map(BasicBlock::new).collect();
    let mut used_blocks = 0;
    for alive_index in seen.iter() {
        replacements[alive_index] = BasicBlock::new(used_blocks);
        if alive_index != used_blocks {
            // Swap the next alive block data with the current available slot. Since alive_index is
            // non-decreasing this is a valid operation.
            basic_blocks.raw.swap(alive_index, used_blocks);
        }
        used_blocks += 1;
    }
    basic_blocks.raw.truncate(used_blocks);

    for block in basic_blocks {
        for target in block.terminator_mut().successors_mut() {
            *target = replacements[target.index()];
        }
    }
}
