//! A pass that propagates the unreachable terminator of a block to its predecessors
//! when all of their successors are unreachable. This is achieved through a
//! post-order traversal of the blocks.

use crate::transform::simplify;
use crate::transform::{MirPass, MirSource};
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use std::borrow::Cow;

pub struct UnreachablePropagation;

impl MirPass<'_> for UnreachablePropagation {
    fn run_pass<'tcx>(&self, tcx: TyCtxt<'tcx>, _: MirSource<'tcx>, body: &mut Body<'tcx>) {
        if tcx.sess.opts.debugging_opts.mir_opt_level < 3 {
            // Enable only under -Zmir-opt-level=3 as in some cases (check the deeply-nested-opt
            // perf benchmark) LLVM may spend quite a lot of time optimizing the generated code.
            return;
        }

        let mut unreachable_blocks = FxHashSet::default();
        let mut replacements = FxHashMap::default();

        for (bb, bb_data) in traversal::postorder(body) {
            let terminator = bb_data.terminator();
            // HACK: If the block contains any asm statement it is not regarded as unreachable.
            // This is a temporary solution that handles possibly diverging asm statements.
            // Accompanying testcases: mir-opt/unreachable_asm.rs and mir-opt/unreachable_asm_2.rs
            let asm_stmt_in_block = || {
                bb_data.statements.iter().any(|stmt: &Statement<'_>| match stmt.kind {
                    StatementKind::LlvmInlineAsm(..) => true,
                    _ => false,
                })
            };

            if terminator.kind == TerminatorKind::Unreachable && !asm_stmt_in_block() {
                unreachable_blocks.insert(bb);
            } else {
                let is_unreachable = |succ: BasicBlock| unreachable_blocks.contains(&succ);
                let terminator_kind_opt = remove_successors(&terminator.kind, is_unreachable);

                if let Some(terminator_kind) = terminator_kind_opt {
                    if terminator_kind == TerminatorKind::Unreachable && !asm_stmt_in_block() {
                        unreachable_blocks.insert(bb);
                    }
                    replacements.insert(bb, terminator_kind);
                }
            }
        }

        let replaced = !replacements.is_empty();
        for (bb, terminator_kind) in replacements {
            body.basic_blocks_mut()[bb].terminator_mut().kind = terminator_kind;
        }

        if replaced {
            simplify::remove_dead_blocks(body);
        }
    }
}

fn remove_successors<F>(
    terminator_kind: &TerminatorKind<'tcx>,
    predicate: F,
) -> Option<TerminatorKind<'tcx>>
where
    F: Fn(BasicBlock) -> bool,
{
    let terminator = match *terminator_kind {
        TerminatorKind::Goto { target } if predicate(target) => TerminatorKind::Unreachable,
        TerminatorKind::SwitchInt { ref discr, switch_ty, ref values, ref targets } => {
            let original_targets_len = targets.len();
            let (otherwise, targets) = targets.split_last().unwrap();
            let (mut values, mut targets): (Vec<_>, Vec<_>) =
                values.iter().zip(targets.iter()).filter(|(_, &t)| !predicate(t)).unzip();

            if !predicate(*otherwise) {
                targets.push(*otherwise);
            } else {
                values.pop();
            }

            let retained_targets_len = targets.len();

            if targets.is_empty() {
                TerminatorKind::Unreachable
            } else if targets.len() == 1 {
                TerminatorKind::Goto { target: targets[0] }
            } else if original_targets_len != retained_targets_len {
                TerminatorKind::SwitchInt {
                    discr: discr.clone(),
                    switch_ty,
                    values: Cow::from(values),
                    targets,
                }
            } else {
                return None;
            }
        }
        _ => return None,
    };
    Some(terminator)
}
