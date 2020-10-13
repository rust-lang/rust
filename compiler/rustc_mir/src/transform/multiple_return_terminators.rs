//! This pass removes jumps to basic blocks containing only a return, and replaces them with a
//! return instead.

use crate::transform::{simplify, MirPass};
use rustc_data_structures::statistics::Statistic;
use rustc_index::bit_set::BitSet;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;

pub struct MultipleReturnTerminators;

static NUM_REPLACED: Statistic =
    Statistic::new(module_path!(), "Number of goto terminators replaced with a return");

impl<'tcx> MirPass<'tcx> for MultipleReturnTerminators {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        if tcx.sess.opts.debugging_opts.mir_opt_level < 3 {
            return;
        }

        // find basic blocks with no statement and a return terminator
        let mut bbs_simple_returns = BitSet::new_empty(body.basic_blocks().len());
        let bbs = body.basic_blocks_mut();
        for idx in bbs.indices() {
            if bbs[idx].statements.is_empty()
                && bbs[idx].terminator().kind == TerminatorKind::Return
            {
                bbs_simple_returns.insert(idx);
            }
        }

        for bb in bbs {
            if let TerminatorKind::Goto { target } = bb.terminator().kind {
                if bbs_simple_returns.contains(target) {
                    NUM_REPLACED.increment(1);
                    bb.terminator_mut().kind = TerminatorKind::Return;
                }
            }
        }

        simplify::remove_dead_blocks(body)
    }
}
