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
 * codegened to LLVM invoke instructions, because invoke is a block terminator
 * in LLVM so we can't insert any code to handle the call's result into the
 * block that performs the call.
 *
 * This function will break those edges by inserting new blocks along them.
 *
 * NOTE: Simplify CFG will happily undo most of the work this pass does.
 *
 */

impl<'tcx> crate::MirPass<'tcx> for AddCallGuards {
    fn run_pass(&self, _tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let mut pred_count = IndexVec::from_elem(0u8, &body.basic_blocks);
        for (_, data) in body.basic_blocks.iter_enumerated() {
            for succ in data.terminator().successors() {
                pred_count[succ] = pred_count[succ].saturating_add(1);
            }
        }

        // We need a place to store the new blocks generated
        let mut new_blocks = Vec::new();

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

        for block in body.basic_blocks_mut() {
            match block.terminator {
                Some(Terminator {
                    kind: TerminatorKind::Call { target: Some(ref mut destination), unwind, .. },
                    source_info,
                }) if pred_count[*destination] > 1
                    && (generates_invoke(unwind) || self == &AllCallEdges) =>
                {
                    // It's a critical edge, break it
                    *destination = new_block(source_info, block.is_cleanup, *destination);
                }
                Some(Terminator {
                    kind:
                        TerminatorKind::InlineAsm {
                            asm_macro: InlineAsmMacro::Asm,
                            ref mut targets,
                            ref operands,
                            unwind,
                            ..
                        },
                    source_info,
                }) if self == &CriticalCallEdges => {
                    let has_outputs = operands.iter().any(|op| {
                        matches!(op, InlineAsmOperand::InOut { .. } | InlineAsmOperand::Out { .. })
                    });
                    let has_labels =
                        operands.iter().any(|op| matches!(op, InlineAsmOperand::Label { .. }));
                    if has_outputs && (has_labels || generates_invoke(unwind)) {
                        for target in targets.iter_mut() {
                            if pred_count[*target] > 1 {
                                *target = new_block(source_info, block.is_cleanup, *target);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        debug!("Broke {} N edges", new_blocks.len());

        body.basic_blocks_mut().extend(new_blocks);
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
