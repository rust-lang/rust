use crate::MirPass;

use rustc_middle::mir::{BasicBlock, Body, Statement, StatementKind, TerminatorKind};
use rustc_middle::ty::TyCtxt;

use tracing::{info, instrument};

pub struct CtfeLimit;

impl<'tcx> MirPass<'tcx> for CtfeLimit {
    #[instrument(skip(self, _tcx, body))]
    fn run_pass(&self, _tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let doms = body.basic_blocks.dominators();
        //info!("Got body with {} basic blocks: {:#?}", body.basic_blocks.len(), body.basic_blocks);
        //info!("With doms: {doms:?}");

        /*
        for (index, basic_block) in body.basic_blocks.iter().enumerate() {
            info!("bb{index}: {basic_block:#?}")
        }*/
        for (index, basic_block) in body.basic_blocks.iter().enumerate() {
            info!(
                "bb{index} -> successors = {:?}",
                basic_block.terminator().successors().collect::<Vec<BasicBlock>>()
            );
        }
        for (index, basic_block) in body.basic_blocks.iter().enumerate() {
            info!("bb{index} -> unwind = {:?}", basic_block.terminator().unwind())
        }

        let mut dominators = Vec::new();
        for idom in 0..body.basic_blocks.len() {
            let mut nodes = Vec::new();
            for inode in 0..body.basic_blocks.len() {
                let dom = BasicBlock::from_usize(idom);
                let node = BasicBlock::from_usize(inode);
                if doms.is_reachable(dom)
                    && doms.is_reachable(node)
                    && doms.is_dominated_by(node, dom)
                {
                    //info!("{idom} dominates {inode}");
                    nodes.push(true);
                } else {
                    nodes.push(false);
                }
            }
            dominators.push(nodes);
        }
        /*
        for idom in 0..body.basic_blocks.len() {
            print!("{idom} | dom | ");
            for inode in 0..body.basic_blocks.len() {
                if dominators[idom][inode] {
                    print!("{inode} | ");
                } else {
                    print!("  | ");
                }
            }
            print!("\n");
        }
        */

        for (index, basic_block) in body.basic_blocks_mut().iter_mut().enumerate() {
            // info!("bb{index}: {basic_block:#?}");
            //info!("bb{index} -> successors = {:?}", basic_block.terminator().successors().collect::<Vec<BasicBlock>>());
            let is_back_edge_or_fn_call = 'label: {
                match basic_block.terminator().kind {
                    TerminatorKind::Call { .. } => {
                        break 'label true;
                    }
                    _ => (),
                }
                for successor in basic_block.terminator().successors() {
                    let s_index = successor.as_usize();
                    if dominators[s_index][index] {
                        info!("{s_index} to {index} is a loop");
                        break 'label true;
                    }
                }
                false
            };
            if is_back_edge_or_fn_call {
                basic_block.statements.push(Statement {
                    source_info: basic_block.terminator().source_info,
                    kind: StatementKind::ConstEvalCounter,
                });
                info!("New basic block statements vector: {:?}", basic_block.statements);
            }
        }
        info!("With doms: {doms:?}");
    }
}
