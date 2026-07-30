//! This pass removes jumps to basic blocks containing only a return, and replaces them with a
//! return instead.

use rustc_index::bit_set::DenseBitSet;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;

use crate::{PassPolicy, simplify};

pub(super) struct MultipleReturnTerminators;

impl<'tcx> crate::MirPass<'tcx> for MultipleReturnTerminators {
    fn policy(&self, sess: &rustc_session::Session) -> PassPolicy {
        PassPolicy::optimization(sess.mir_opt_level() >= 4)
    }

    fn run_pass(&self, _: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        // find basic blocks with no statement and a return terminator
        let mut bbs_simple_returns = DenseBitSet::new_empty(body.basic_blocks.len());
        let bbs = body.basic_blocks_mut();
        for (idx, bb) in bbs.iter_enumerated() {
            if bb.statements.is_empty() && bb.terminator().kind == TerminatorKind::Return {
                bbs_simple_returns.insert(idx);
            }
        }

        for bb in bbs {
            if let TerminatorKind::Goto { target } = bb.terminator().kind
                && bbs_simple_returns.contains(target)
            {
                bb.terminator_mut().kind = TerminatorKind::Return;
            }
        }

        simplify::remove_dead_blocks(body)
    }
}
