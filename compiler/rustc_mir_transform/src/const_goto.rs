//! This pass optimizes the following sequence
//! ```rust,ignore (example)
//! bb2: {
//!     _2 = const true;
//!     goto -> bb3;
//! }
//!
//! bb3: {
//!     switchInt(_2) -> [false: bb4, otherwise: bb5];
//! }
//! ```
//! into
//! ```rust,ignore (example)
//! bb2: {
//!     _2 = const true;
//!     goto -> bb5;
//! }
//! ```

use crate::MirPass;
use rustc_middle::mir::*;
use rustc_middle::ty::TyCtxt;
use rustc_middle::{mir::visit::Visitor, ty::ParamEnv};

use super::simplify::{simplify_cfg, simplify_locals};

pub struct ConstGoto;

impl<'tcx> MirPass<'tcx> for ConstGoto {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        // This pass participates in some as-of-yet untested unsoundness found
        // in https://github.com/rust-lang/rust/issues/112460
        sess.mir_opt_level() >= 2 && sess.opts.unstable_opts.unsound_mir_opts
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        trace!("Running ConstGoto on {:?}", body.source);
        let param_env = tcx.param_env_reveal_all_normalized(body.source.def_id());
        let mut opt_finder =
            ConstGotoOptimizationFinder { tcx, body, optimizations: vec![], param_env };
        opt_finder.visit_body(body);
        let should_simplify = !opt_finder.optimizations.is_empty();
        for opt in opt_finder.optimizations {
            let block = &mut body.basic_blocks_mut()[opt.bb_with_goto];
            block.statements.extend(opt.stmts_move_up);
            let terminator = block.terminator_mut();
            let new_goto = TerminatorKind::Goto { target: opt.target_to_use_in_goto };
            debug!("SUCCESS: replacing `{:?}` with `{:?}`", terminator.kind, new_goto);
            terminator.kind = new_goto;
        }

        // if we applied optimizations, we potentially have some cfg to cleanup to
        // make it easier for further passes
        if should_simplify {
            simplify_cfg(tcx, body);
            simplify_locals(body, tcx);
        }
    }
}

impl<'tcx> Visitor<'tcx> for ConstGotoOptimizationFinder<'_, 'tcx> {
    fn visit_basic_block_data(&mut self, block: BasicBlock, data: &BasicBlockData<'tcx>) {
        if data.is_cleanup {
            // Because of the restrictions around control flow in cleanup blocks, we don't perform
            // this optimization at all in such blocks.
            return;
        }
        self.super_basic_block_data(block, data);
    }

    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        let _: Option<_> = try {
            let target = terminator.kind.as_goto()?;
            // We only apply this optimization if the last statement is a const assignment
            let last_statement = self.body.basic_blocks[location.block].statements.last()?;

            if let (place, Rvalue::Use(Operand::Constant(_const))) =
                last_statement.kind.as_assign()?
            {
                // We found a constant being assigned to `place`.
                // Now check that the target of this Goto switches on this place.
                let target_bb = &self.body.basic_blocks[target];

                // The `StorageDead(..)` statement does not affect the functionality of mir.
                // We can move this part of the statement up to the predecessor.
                let mut stmts_move_up = Vec::new();
                for stmt in &target_bb.statements {
                    if let StatementKind::StorageDead(..) = stmt.kind {
                        stmts_move_up.push(stmt.clone())
                    } else {
                        None?;
                    }
                }

                let target_bb_terminator = target_bb.terminator();
                let (discr, targets) = target_bb_terminator.kind.as_switch()?;
                if discr.place() == Some(*place) {
                    let switch_ty = place.ty(self.body.local_decls(), self.tcx).ty;
                    // We now know that the Switch matches on the const place, and it is statementless
                    // Now find which value in the Switch matches the const value.
                    let const_value =
                        _const.literal.try_eval_bits(self.tcx, self.param_env, switch_ty)?;
                    let target_to_use_in_goto = targets.target_for_value(const_value);
                    self.optimizations.push(OptimizationToApply {
                        bb_with_goto: location.block,
                        target_to_use_in_goto,
                        stmts_move_up,
                    });
                }
            }
            Some(())
        };

        self.super_terminator(terminator, location);
    }
}

struct OptimizationToApply<'tcx> {
    bb_with_goto: BasicBlock,
    target_to_use_in_goto: BasicBlock,
    stmts_move_up: Vec<Statement<'tcx>>,
}

pub struct ConstGotoOptimizationFinder<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    param_env: ParamEnv<'tcx>,
    optimizations: Vec<OptimizationToApply<'tcx>>,
}
