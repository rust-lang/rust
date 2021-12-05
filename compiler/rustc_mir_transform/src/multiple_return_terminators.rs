//! This pass removes jumps to basic blocks containing only a return, and replaces them with a
//! return instead.

use crate::{simplify, MirPass};
use rustc_index::bit_set::BitSet;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;

pub struct MultipleReturnTerminators;

impl<'tcx> MirPass<'tcx> for MultipleReturnTerminators {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() >= 4
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        // find basic blocks with no statement and a return terminator
        let mut bbs_simple_returns = BitSet::new_empty(body.basic_blocks().len());
        let def_id = body.source.def_id();
        let bbs = body.basic_blocks_mut();
        for idx in bbs.indices() {
            if bbs[idx].statements.is_empty()
                && bbs[idx].terminator().kind == TerminatorKind::Return
            {
                bbs_simple_returns.insert(idx);
            }
        }

        for bb in bbs {
            if !tcx.consider_optimizing(|| format!("MultipleReturnTerminators {:?} ", def_id)) {
                break;
            }

            if let TerminatorKind::Goto { target } = bb.terminator().kind {
                if bbs_simple_returns.contains(target) {
                    bb.terminator_mut().kind = TerminatorKind::Return;
                }
            }
        }

        simplify::remove_dead_blocks(tcx, body)
    }
}
