//! This module contains the `EvalContext` methods for executing a single step of the interpreter.
//!
//! The main entry point is the `step` method.

use rustc::mir;
use rustc::lint;

use rustc::mir::interpret::EvalResult;
use super::{EvalContext, Machine};

impl<'a, 'mir, 'tcx, M: Machine<'mir, 'tcx>> EvalContext<'a, 'mir, 'tcx, M> {
    pub fn inc_step_counter_and_check_limit(&mut self, n: usize) {
        self.terminators_remaining = self.terminators_remaining.saturating_sub(n);
        if self.terminators_remaining == 0 {
            let msg = "constant evaluating a complex constant, this might take some time";
            let lint = ::rustc::lint::builtin::CONST_TIME_LIMIT;
            let node_id = self
                .stack()
                .iter()
                .rev()
                .filter_map(|frame| frame.lint_root)
                .next()
                .or(lint_root)
                .expect("some part of a failing const eval must be local");
            let err = self.tcx.struct_span_lint_node(
                lint,
                node_id,
                self.frame().span,
                msg,
            );
            if self.lint_level_at_node(lint, node_id).0 == lint::Level::Deny ||
                self.lint_level_at_node(lint, node_id).0 == lint::Level::Forbid {
                    err.report_as_err(self.tcx, msg, node_id);
                } else {
                    err.report_as_lint(self.tcx, msg);
                }
            self.terminators_remaining = 1_000_000;
        }
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

        self.inc_step_counter_and_check_limit(1);

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
