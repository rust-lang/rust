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
        let mut pred_count: IndexVec<_, _> =
            body.basic_blocks.predecessors().iter().map(|ps| ps.len()).collect();
        pred_count[START_BLOCK] += 1;

        // We need a place to store the new blocks generated
        let mut new_blocks = Vec::new();

        let cur_len = body.basic_blocks.len();

        for block in body.basic_blocks_mut() {
            match block.terminator {
                Some(Terminator {
                    kind: TerminatorKind::Call { target: Some(ref mut destination), unwind, .. },
                    source_info,
                }) if pred_count[*destination] > 1
                    && (matches!(
                        unwind,
                        UnwindAction::Cleanup(_) | UnwindAction::Terminate(_)
                    ) || self == &AllCallEdges) =>
                {
                    // It's a critical edge, break it
                    let call_guard = BasicBlockData {
                        statements: vec![],
                        is_cleanup: block.is_cleanup,
                        terminator: Some(Terminator {
                            source_info,
                            kind: TerminatorKind::Goto { target: *destination },
                        }),
                    };

                    // Get the index it will be when inserted into the MIR
                    let idx = cur_len + new_blocks.len();
                    new_blocks.push(call_guard);
                    *destination = BasicBlock::new(idx);
                }
                _ => {}
            }
        }

        debug!("Broke {} N edges", new_blocks.len());

        body.basic_blocks_mut().extend(new_blocks);
    }
}
