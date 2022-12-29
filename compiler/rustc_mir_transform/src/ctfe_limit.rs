use crate::MirPass;

use rustc_middle::mir::{
    BasicBlock, BasicBlockData, Body, Statement, StatementKind, TerminatorKind,
};
use rustc_middle::ty::TyCtxt;

pub struct CtfeLimit;

impl<'tcx> MirPass<'tcx> for CtfeLimit {
    #[instrument(skip(self, _tcx, body))]
    fn run_pass(&self, _tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let doms = body.basic_blocks.dominators();
        let indices: Vec<BasicBlock> =
            body.basic_blocks
                .iter_enumerated()
                .filter_map(|(node, node_data)| {
                    if matches!(node_data.terminator().kind, TerminatorKind::Call { .. }) ||
                    // Back edges in a CFG indicate loops
                    body.basic_blocks.iter_enumerated().any(|(potential_dom, _)| {
                        doms.is_reachable(potential_dom)
                            && doms.is_reachable(node)
                            && doms.is_dominated_by(node, potential_dom)
                            && node_data
                                .terminator()
                                .successors()
                                .into_iter()
                                .any(|succ| succ == potential_dom)
                    }) {
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

fn insert_counter(basic_block_data: &mut BasicBlockData<'_>) {
    basic_block_data.statements.push(Statement {
        source_info: basic_block_data.terminator().source_info,
        kind: StatementKind::ConstEvalCounter,
    });
}
