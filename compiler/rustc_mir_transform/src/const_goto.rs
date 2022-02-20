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
        sess.mir_opt_level() >= 4
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        trace!("Running ConstGoto on {:?}", body.source);
        let param_env = tcx.param_env_reveal_all_normalized(body.source.def_id());
        let mut opt_finder =
            ConstGotoOptimizationFinder { tcx, body, optimizations: vec![], param_env };
        opt_finder.visit_body(body);
        let should_simplify = !opt_finder.optimizations.is_empty();
        for opt in opt_finder.optimizations {
            let terminator = body.basic_blocks_mut()[opt.bb_with_goto].terminator_mut();
            let new_goto = TerminatorKind::Goto { target: opt.target_to_use_in_goto };
            debug!("SUCCESS: replacing `{:?}` with `{:?}`", terminator.kind, new_goto);
            terminator.kind = new_goto;

            let bridge_bb_statement = &body.basic_blocks()[opt.bridge_bb].statements;
            if !bridge_bb_statement.is_empty() {
                let storagedeads = bridge_bb_statement.clone();
                body.basic_blocks_mut()[opt.bb_with_goto].statements.extend(storagedeads);
                debug!("SUCCESS: extending terminator's basic block with briding basic block.");
            }
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
    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        let _: Option<_> = try {
            let target = terminator.kind.as_goto()?;
            // We only apply this optimization if the last statement is a const assignment
            let last_statement = self.body.basic_blocks()[location.block].statements.last()?;

            if let (place, Rvalue::Use(Operand::Constant(_const))) =
                last_statement.kind.as_assign()?
            {
                // We found a constant being assigned to `place`.
                // Now check that the target of this Goto switches on this place.
                let target_bb = &self.body.basic_blocks()[target];
                let target_bb_statements = &target_bb.statements;
                if !target_bb_statements.is_empty()
                    && !target_bb_statements.iter().all(|statement| match statement.kind {
                        StatementKind::StorageDead(_) => true,
                        _ => false,
                    })
                {
                    None?
                }

                let target_bb_terminator = target_bb.terminator();
                let (discr, switch_ty, targets) = target_bb_terminator.kind.as_switch()?;
                if discr.place() == Some(*place) {
                    // We now know that the Switch matches on the const place, and it is statementless
                    // Now find which value in the Switch matches the const value.
                    let const_value =
                        _const.literal.try_eval_bits(self.tcx, self.param_env, switch_ty)?;
                    let target_to_use_in_goto = targets.target_for_value(const_value);
                    self.optimizations.push(OptimizationToApply {
                        bb_with_goto: location.block,
                        bridge_bb: target,
                        target_to_use_in_goto,
                    });
                }
            }
            Some(())
        };

        self.super_terminator(terminator, location);
    }
}

struct OptimizationToApply {
    bb_with_goto: BasicBlock,
    bridge_bb: BasicBlock,
    target_to_use_in_goto: BasicBlock,
}

pub struct ConstGotoOptimizationFinder<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    param_env: ParamEnv<'tcx>,
    optimizations: Vec<OptimizationToApply>,
}
