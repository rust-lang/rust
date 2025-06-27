use rustc_abi::ExternAbi;

use self::shims::windows::handle::{EvalContextExt as _, Handle, PseudoHandle};
use crate::*;

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}

#[allow(non_snake_case)]
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn CreateThread(
        &mut self,
        security_op: &OpTy<'tcx>,
        stacksize_op: &OpTy<'tcx>,
        start_op: &OpTy<'tcx>,
        arg_op: &OpTy<'tcx>,
        flags_op: &OpTy<'tcx>,
        thread_op: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, ThreadId> {
        let this = self.eval_context_mut();

        let security = this.read_pointer(security_op)?;
        // stacksize is ignored, but still needs to be a valid usize
        this.read_target_usize(stacksize_op)?;
        let start_routine = this.read_pointer(start_op)?;
        let func_arg = this.read_immediate(arg_op)?;
        let flags = this.read_scalar(flags_op)?.to_u32()?;

        let thread = if this.ptr_is_null(this.read_pointer(thread_op)?)? {
            None
        } else {
            let thread_info_place = this.deref_pointer_as(thread_op, this.machine.layouts.u32)?;
            Some(thread_info_place)
        };

        let stack_size_param_is_a_reservation =
            this.eval_windows_u32("c", "STACK_SIZE_PARAM_IS_A_RESERVATION");

        // We ignore the stack size, so we also ignore the
        // `STACK_SIZE_PARAM_IS_A_RESERVATION` flag.
        if flags != 0 && flags != stack_size_param_is_a_reservation {
            throw_unsup_format!("unsupported `dwCreationFlags` {} in `CreateThread`", flags)
        }

        if !this.ptr_is_null(security)? {
            throw_unsup_format!("non-null `lpThreadAttributes` in `CreateThread`")
        }

        this.start_regular_thread(
            thread,
            start_routine,
            ExternAbi::System { unwind: false },
            func_arg,
            this.layout_of(this.tcx.types.u32)?,
        )
    }

    fn WaitForSingleObject(
        &mut self,
        handle_op: &OpTy<'tcx>,
        timeout_op: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let handle = this.read_handle(handle_op, "WaitForSingleObject")?;
        let timeout = this.read_scalar(timeout_op)?.to_u32()?;

        let thread = match handle {
            Handle::Thread(thread) => thread,
            // Unlike on posix, the outcome of joining the current thread is not documented.
            // On current Windows, it just deadlocks.
            Handle::Pseudo(PseudoHandle::CurrentThread) => this.active_thread(),
            _ => this.invalid_handle("WaitForSingleObject")?,
        };

        if timeout != this.eval_windows_u32("c", "INFINITE") {
            throw_unsup_format!("`WaitForSingleObject` with non-infinite timeout");
        }

        this.join_thread(thread)?;

        interp_ok(this.eval_windows("c", "WAIT_OBJECT_0"))
    }
}
