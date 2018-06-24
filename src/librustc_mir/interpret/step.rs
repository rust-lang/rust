//! This module contains the `EvalContext` methods for executing a single step of the interpreter.
//!
//! The main entry point is the `step` method.

use std::hash::Hash;

use rustc::mir;

use rustc::mir::interpret::EvalResult;
use super::{EvalContext, Machine};

impl<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> EvalContext<'a, 'mir, 'tcx, M>
    where M: Clone + Eq + Hash,
{
    /// Returns `true` if the loop detector should take a snapshot during the current step.
    pub fn is_loop_detector_scheduled(&self) -> bool {
        /// The number of steps between loop detector snapshots.
        /// Should be a power of two for performance reasons.
        const LOOP_SNAPSHOT_PERIOD: isize = 1 << 8;

        let steps = self.steps_until_detector_enabled;
        steps <= 0 && steps % LOOP_SNAPSHOT_PERIOD == 0
    }

    pub fn inc_step_counter_and_detect_loops(&mut self, n: usize) -> EvalResult<'tcx, ()> {
        // TODO: Remove `as` cast
        self.steps_until_detector_enabled =
            self.steps_until_detector_enabled.saturating_sub(n as isize);

        if !self.is_loop_detector_scheduled() {
            return Ok(());
        }

        if self.loop_detector.is_empty() {
            // First run of the loop detector

            // FIXME(#49980): make this warning a lint
            self.tcx.sess.span_warn(self.frame().span,
                "Constant evaluating a complex constant, this might take some time");
        }

        self.loop_detector.observe_and_analyze(&self.machine, &self.stack, &self.memory)
    }

    /// Returns true as long as there are more things to do.
    pub fn step(&mut self) -> EvalResult<'tcx, bool> {
        if self.stack.is_empty() {
            return Ok(false);
        }

        let block = self.frame().block;
        let stmt_id = self.frame().stmt;
        let mir = self.mir();
        let basic_block = &mir.basic_blocks()[block];

        let old_frames = self.cur_frame();

        if let Some(stmt) = basic_block.statements.get(stmt_id) {
            assert_eq!(old_frames, self.cur_frame());
            self.statement(stmt)?;
            return Ok(true);
        }

        self.inc_step_counter_and_detect_loops(1)?;

        let terminator = basic_block.terminator();
        assert_eq!(old_frames, self.cur_frame());
        self.terminator(terminator)?;
        Ok(true)
    }

    fn statement(&mut self, stmt: &mir::Statement<'tcx>) -> EvalResult<'tcx> {
        trace!("{:?}", stmt);

        use rustc::mir::StatementKind::*;

        // Some statements (e.g. box) push new stack frames.  We have to record the stack frame number
        // *before* executing the statement.
        let frame_idx = self.cur_frame();
        self.tcx.span = stmt.source_info.span;
        self.memory.tcx.span = stmt.source_info.span;

        match stmt.kind {
            Assign(ref place, ref rvalue) => self.eval_rvalue_into_place(rvalue, place)?,

            SetDiscriminant {
                ref place,
                variant_index,
            } => {
                let dest = self.eval_place(place)?;
                let dest_ty = self.place_ty(place);
                self.write_discriminant_value(dest_ty, dest, variant_index)?;
            }

            // Mark locals as alive
            StorageLive(local) => {
                let old_val = self.frame_mut().storage_live(local);
                self.deallocate_local(old_val)?;
            }

            // Mark locals as dead
            StorageDead(local) => {
                let old_val = self.frame_mut().storage_dead(local);
                self.deallocate_local(old_val)?;
            }

            // No dynamic semantics attached to `ReadForMatch`; MIR
            // interpreter is solely intended for borrowck'ed code.
            ReadForMatch(..) => {}

            // Validity checks.
            Validate(op, ref places) => {
                for operand in places {
                    M::validation_op(self, op, operand)?;
                }
            }
            EndRegion(ce) => {
                M::end_region(self, Some(ce))?;
            }

            UserAssertTy(..) => {}

            // Defined to do nothing. These are added by optimization passes, to avoid changing the
            // size of MIR constantly.
            Nop => {}

            InlineAsm { .. } => return err!(InlineAsm),
        }

        self.stack[frame_idx].stmt += 1;
        Ok(())
    }

    fn terminator(&mut self, terminator: &mir::Terminator<'tcx>) -> EvalResult<'tcx> {
        trace!("{:?}", terminator.kind);
        self.tcx.span = terminator.source_info.span;
        self.memory.tcx.span = terminator.source_info.span;
        self.eval_terminator(terminator)?;
        if !self.stack.is_empty() {
            trace!("// {:?}", self.frame().block);
        }
        Ok(())
    }
}
