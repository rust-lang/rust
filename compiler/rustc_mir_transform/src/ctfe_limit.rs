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
                if matches!(node_data.terminator().kind, TerminatorKind::Call { .. })
                    // Back edges in a CFG indicate loops
                    || has_back_edge(doms, node, node_data)
                {
                    Some(node)
                } else {
                    None
                }
            })
            .collect();
        for index in indices {
            insert_counter(
                body.basic_blocks_mut()
                    .get_mut(index)
                    .expect("basic_blocks index {index} should exist"),
            );
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

fn insert_counter(basic_block_data: &mut BasicBlockData<'_>) {
    basic_block_data.statements.push(Statement::new(
        basic_block_data.terminator().source_info,
        StatementKind::ConstEvalCounter,
    ));
}
