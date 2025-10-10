//! A pass that inserts the `ConstEvalCounter` instruction into any blocks that have a back edge
//! (thus indicating there is a loop in the CFG), or whose terminator is a function call.

use rustc_data_structures::graph::dominators::Dominators;
use rustc_middle::mir::{
    BasicBlock, BasicBlockData, Body, Statement, StatementKind, TerminatorKind,
};
use rustc_middle::ty::TyCtxt;
use tracing::instrument;

pub(super) struct CtfeLimit;

impl<'tcx> crate::MirPass<'tcx> for CtfeLimit {
    #[instrument(skip(self, _tcx, body))]
    fn run_pass(&self, _tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let doms = body.basic_blocks.dominators();
        let indices: Vec<BasicBlock> = body
            .basic_blocks
            .iter_enumerated()
            .filter_map(|(node, node_data)| {
                if matches!(node_data.terminator().kind, TerminatorKind::Call { .. } | TerminatorKind::TailCall { .. })
                    // Back edges in a CFG indicate loops
                    || has_back_edge(doms, node, node_data)
                {
                    Some(node)
                } else {
                    None
                }
            })
            .collect();

        let basic_blocks = body.basic_blocks.as_mut_preserves_cfg();
        for index in indices {
            let bbdata = &mut basic_blocks[index];
            let source_info = bbdata.terminator().source_info;
            bbdata.statements.push(Statement::new(source_info, StatementKind::ConstEvalCounter));
        }
    }

    fn is_required(&self) -> bool {
        true
    }
}

fn has_back_edge(
    doms: &Dominators<BasicBlock>,
    node: BasicBlock,
    node_data: &BasicBlockData<'_>,
) -> bool {
    if !doms.is_reachable(node) {
        return false;
    }
    // Check if any of the dominators of the node are also the node's successor.
    node_data.terminator().successors().any(|succ| doms.dominates(succ, node))
}
