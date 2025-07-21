//! Helper functions for causing panics.

use rustc_abi::ExternAbi;
use rustc_middle::{mir, ty};

use crate::*;

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Start a panic in the interpreter with the given message as payload.
    fn start_panic(&mut self, msg: &str, unwind: mir::UnwindAction) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        // First arg: message.
        let msg = this.allocate_str_dedup(msg)?;

        // Call the lang item.
        let panic = this.tcx.lang_items().panic_fn().unwrap();
        let panic = ty::Instance::mono(this.tcx.tcx, panic);
        this.call_function(
            panic,
            ExternAbi::Rust,
            &[this.mplace_to_ref(&msg)?],
            None,
            ReturnContinuation::Goto { ret: None, unwind },
        )
    }

    /// Start a non-unwinding panic in the interpreter with the given message as payload.
    fn start_panic_nounwind(&mut self, msg: &str) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        // First arg: message.
        let msg = this.allocate_str_dedup(msg)?;

        // Call the lang item.
        let panic = this.tcx.lang_items().panic_nounwind().unwrap();
        let panic = ty::Instance::mono(this.tcx.tcx, panic);
        this.call_function(
            panic,
            ExternAbi::Rust,
            &[this.mplace_to_ref(&msg)?],
            None,
            ReturnContinuation::Goto { ret: None, unwind: mir::UnwindAction::Unreachable },
        )
    }

    fn assert_panic(
        &mut self,
        msg: &mir::AssertMessage<'tcx>,
        unwind: mir::UnwindAction,
    ) -> InterpResult<'tcx> {
        use rustc_middle::mir::AssertKind::*;
        let this = self.eval_context_mut();

        match msg {
            BoundsCheck { index, len } => {
                // Forward to `panic_bounds_check` lang item.

                // First arg: index.
                let index = this.read_immediate(&this.eval_operand(index, None)?)?;
                // Second arg: len.
                let len = this.read_immediate(&this.eval_operand(len, None)?)?;

                // Call the lang item.
                let panic_bounds_check = this.tcx.lang_items().panic_bounds_check_fn().unwrap();
                let panic_bounds_check = ty::Instance::mono(this.tcx.tcx, panic_bounds_check);
                this.call_function(
                    panic_bounds_check,
                    ExternAbi::Rust,
                    &[index, len],
                    None,
                    ReturnContinuation::Goto { ret: None, unwind },
                )?;
            }
            MisalignedPointerDereference { required, found } => {
                // Forward to `panic_misaligned_pointer_dereference` lang item.

                // First arg: required.
                let required = this.read_immediate(&this.eval_operand(required, None)?)?;
                // Second arg: found.
                let found = this.read_immediate(&this.eval_operand(found, None)?)?;

                // Call the lang item.
                let panic_misaligned_pointer_dereference =
                    this.tcx.lang_items().panic_misaligned_pointer_dereference_fn().unwrap();
                let panic_misaligned_pointer_dereference =
                    ty::Instance::mono(this.tcx.tcx, panic_misaligned_pointer_dereference);
                this.call_function(
                    panic_misaligned_pointer_dereference,
                    ExternAbi::Rust,
                    &[required, found],
                    None,
                    ReturnContinuation::Goto { ret: None, unwind },
                )?;
            }

            _ => {
                // Call the lang item associated with this message.
                let fn_item = this.tcx.require_lang_item(msg.panic_function(), this.tcx.span);
                let instance = ty::Instance::mono(this.tcx.tcx, fn_item);
                this.call_function(
                    instance,
                    ExternAbi::Rust,
                    &[],
                    None,
                    ReturnContinuation::Goto { ret: None, unwind },
                )?;
            }
        }
        interp_ok(())
    }
}
