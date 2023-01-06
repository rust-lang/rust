//! A pass that inserts the `ConstEvalCounter` instruction into any blocks that have a back edge
//! (thus indicating there is a loop in the CFG), or whose terminator is a function call.
use crate::MirPass;

use rustc_middle::mir::{
    BasicBlock, BasicBlockData, BasicBlocks, Body, Statement, StatementKind, TerminatorKind,
};
use rustc_middle::ty::TyCtxt;

pub struct CtfeLimit;

impl<'tcx> MirPass<'tcx> for CtfeLimit {
    #[instrument(skip(self, _tcx, body))]
    fn run_pass(&self, _tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let indices: Vec<BasicBlock> = body
            .basic_blocks
            .iter_enumerated()
            .filter_map(|(node, node_data)| {
                if matches!(node_data.terminator().kind, TerminatorKind::Call { .. })
                    // Back edges in a CFG indicate loops
                    || has_back_edge(&body.basic_blocks, node, &node_data)
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
}

fn has_back_edge(
    basic_blocks: &BasicBlocks<'_>,
    node: BasicBlock,
    node_data: &BasicBlockData<'_>,
) -> bool {
    let doms = basic_blocks.dominators();
    basic_blocks.indices().any(|potential_dom| {
        doms.is_reachable(potential_dom)
            && doms.is_reachable(node)
            && doms.is_dominated_by(node, potential_dom)
            && node_data.terminator().successors().into_iter().any(|succ| succ == potential_dom)
    })
}

fn insert_counter(basic_block_data: &mut BasicBlockData<'_>) {
    basic_block_data.statements.push(Statement {
        source_info: basic_block_data.terminator().source_info,
        kind: StatementKind::ConstEvalCounter,
    });
}
