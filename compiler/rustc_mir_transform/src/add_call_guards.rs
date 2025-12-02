//! Breaks outgoing critical edges for call terminators in the MIR.
//!
//! Critical edges are edges that are neither the only edge leaving a
//! block, nor the only edge entering one.
//!
//! When you want something to happen "along" an edge, you can either
//! do at the end of the predecessor block, or at the start of the
//! successor block. Critical edges have to be broken in order to prevent
//! "edge actions" from affecting other edges. We need this for calls that are
//! codegened to LLVM invoke instructions, because invoke is a block terminator
//! in LLVM so we can't insert any code to handle the call's result into the
//! block that performs the call.
//!
//! This function will break those edges by inserting new blocks along them.
//!
//! NOTE: Simplify CFG will happily undo most of the work this pass does.

use rustc_index::{Idx, IndexVec};
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use tracing::debug;

#[derive(PartialEq)]
pub(super) enum AddCallGuards {
    AllCallEdges,
    CriticalCallEdges,
}
pub(super) use self::AddCallGuards::*;

impl<'tcx> crate::MirPass<'tcx> for AddCallGuards {
    fn run_pass(&self, _tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let mut pred_count = IndexVec::from_elem(0u8, &body.basic_blocks);
        for (_, data) in body.basic_blocks.iter_enumerated() {
            for succ in data.terminator().successors() {
                pred_count[succ] = pred_count[succ].saturating_add(1);
            }
        }

        enum Action {
            Call,
            Asm { target_index: usize },
        }

        let mut work = Vec::with_capacity(body.basic_blocks.len());
        for (bb, block) in body.basic_blocks.iter_enumerated() {
            let term = block.terminator();
            match term.kind {
                TerminatorKind::Call { target: Some(destination), unwind, .. }
                    if pred_count[destination] > 1
                        && (generates_invoke(unwind) || self == &AllCallEdges) =>
                {
                    // It's a critical edge, break it
                    work.push((bb, Action::Call));
                }
                TerminatorKind::InlineAsm {
                    asm_macro: InlineAsmMacro::Asm,
                    ref targets,
                    ref operands,
                    unwind,
                    ..
                } if self == &CriticalCallEdges => {
                    let has_outputs = operands.iter().any(|op| {
                        matches!(op, InlineAsmOperand::InOut { .. } | InlineAsmOperand::Out { .. })
                    });
                    let has_labels =
                        operands.iter().any(|op| matches!(op, InlineAsmOperand::Label { .. }));
                    if has_outputs && (has_labels || generates_invoke(unwind)) {
                        for (target_index, target) in targets.iter().enumerate() {
                            if pred_count[*target] > 1 {
                                work.push((bb, Action::Asm { target_index }));
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        if work.is_empty() {
            return;
        }

        // We need a place to store the new blocks generated
        let mut new_blocks = Vec::with_capacity(work.len());

        let cur_len = body.basic_blocks.len();
        let mut new_block = |source_info: SourceInfo, is_cleanup: bool, target: BasicBlock| {
            let block = BasicBlockData::new(
                Some(Terminator { source_info, kind: TerminatorKind::Goto { target } }),
                is_cleanup,
            );
            let idx = cur_len + new_blocks.len();
            new_blocks.push(block);
            BasicBlock::new(idx)
        };

        let basic_blocks = body.basic_blocks.as_mut();
        for (source, action) in work {
            let block = &mut basic_blocks[source];
            let is_cleanup = block.is_cleanup;
            let term = block.terminator_mut();
            let source_info = term.source_info;
            let destination = match action {
                Action::Call => {
                    let TerminatorKind::Call { target: Some(ref mut destination), .. } = term.kind
                    else {
                        unreachable!()
                    };
                    destination
                }
                Action::Asm { target_index } => {
                    let TerminatorKind::InlineAsm { ref mut targets, .. } = term.kind else {
                        unreachable!()
                    };
                    &mut targets[target_index]
                }
            };
            *destination = new_block(source_info, is_cleanup, *destination);
        }

        debug!("Broke {} N edges", new_blocks.len());
        basic_blocks.extend(new_blocks);
    }

    fn is_required(&self) -> bool {
        true
    }
}

/// Returns true if this unwind action is code generated as an invoke as opposed to a call.
fn generates_invoke(unwind: UnwindAction) -> bool {
    match unwind {
        UnwindAction::Continue | UnwindAction::Unreachable => false,
        UnwindAction::Cleanup(_) | UnwindAction::Terminate(_) => true,
    }
}
