use rustc::ty::TyCtxt;
use rustc::mir::*;
use rustc_data_structures::indexed_vec::{Idx, IndexVec};
use crate::transform::{MirPass, MirSource};

#[derive(PartialEq)]
pub enum AddCallGuards {
    AllCallEdges,
    CriticalCallEdges,
}
pub use self::AddCallGuards::*;

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

impl MirPass for AddCallGuards {
    fn run_pass<'tcx>(&self, _tcx: TyCtxt<'tcx>, _src: MirSource<'tcx>, body: &mut Body<'tcx>) {
        self.add_call_guards(body);
    }
}

impl AddCallGuards {
    pub fn add_call_guards(&self, body: &mut Body<'_>) {
        let pred_count: IndexVec<_, _> =
            body.predecessors().iter().map(|ps| ps.len()).collect();

        // We need a place to store the new blocks generated
        let mut new_blocks = Vec::new();

        let cur_len = body.basic_blocks().len();

        for block in body.basic_blocks_mut() {
            match block.terminator {
                Some(Terminator {
                    kind: TerminatorKind::Call {
                        destination: Some((_, ref mut destination)),
                        cleanup,
                        ..
                    }, source_info
                }) if pred_count[*destination] > 1 &&
                      (cleanup.is_some() || self == &AllCallEdges) =>
                {
                    // It's a critical edge, break it
                    let call_guard = BasicBlockData {
                        statements: vec![],
                        is_cleanup: block.is_cleanup,
                        terminator: Some(Terminator {
                            source_info,
                            kind: TerminatorKind::Goto { target: *destination }
                        })
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
