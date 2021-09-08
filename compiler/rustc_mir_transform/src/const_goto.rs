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
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        if tcx.sess.mir_opt_level() < 4 {
            return;
        }
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
        }

        // if we applied optimizations, we potentially have some cfg to cleanup to
        // make it easier for further passes
        if should_simplify {
            simplify_cfg(tcx, body);
            simplify_locals(body, tcx);
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for ConstGotoOptimizationFinder<'a, 'tcx> {
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

                // FIXME(simonvandel): We are conservative here when we don't allow
                // any statements in the target basic block.
                // This could probably be relaxed to allow `StorageDead`s which could be
                // copied to the predecessor of this block.
                if !target_bb.statements.is_empty() {
                    None?
                }

                let target_bb_terminator = target_bb.terminator();
                let (discr, switch_ty, targets) = target_bb_terminator.kind.as_switch()?;
                if discr.place() == Some(*place) {
                    // We now know that the Switch matches on the const place, and it is statementless
                    // Now find which value in the Switch matches the const value.
                    let const_value =
                        _const.literal.try_eval_bits(self.tcx, self.param_env, switch_ty)?;
                    let found_value_idx_option = targets
                        .iter()
                        .enumerate()
                        .find(|(_, (value, _))| const_value == *value)
                        .map(|(idx, _)| idx);

                    let target_to_use_in_goto =
                        if let Some(found_value_idx) = found_value_idx_option {
                            targets.iter().nth(found_value_idx).unwrap().1
                        } else {
                            // If we did not find the const value in values, it must be the otherwise case
                            targets.otherwise()
                        };

                    self.optimizations.push(OptimizationToApply {
                        bb_with_goto: location.block,
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
    target_to_use_in_goto: BasicBlock,
}

pub struct ConstGotoOptimizationFinder<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
    param_env: ParamEnv<'tcx>,
    optimizations: Vec<OptimizationToApply>,
}
