use rustc_middle::ty::layout::LayoutOf;
use rustc_target::spec::abi::Abi;

use crate::*;
use shims::windows::handle::{EvalContextExt as _, Handle, PseudoHandle};

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}

#[allow(non_snake_case)]
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn CreateThread(
        &mut self,
        security_op: &OpTy<'tcx, Provenance>,
        stacksize_op: &OpTy<'tcx, Provenance>,
        start_op: &OpTy<'tcx, Provenance>,
        arg_op: &OpTy<'tcx, Provenance>,
        flags_op: &OpTy<'tcx, Provenance>,
        thread_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, ThreadId> {
        let this = self.eval_context_mut();

        if !this.ptr_is_null(this.read_pointer(security_op)?)? {
            throw_unsup_format!("non-null `lpThreadAttributes` in `CreateThread`")
        }

        // stacksize is ignored, but still needs to be a valid usize
        let _ = this.read_scalar(stacksize_op)?.to_machine_usize(this)?;

        let flags = this.read_scalar(flags_op)?.to_u32()?;

        let stack_size_param_is_a_reservation =
            this.eval_windows("c", "STACK_SIZE_PARAM_IS_A_RESERVATION")?.to_u32()?;

        if flags != 0 && flags != stack_size_param_is_a_reservation {
            throw_unsup_format!("unsupported `dwCreationFlags` {} in `CreateThread`", flags)
        }

        let thread =
            if this.ptr_is_null(this.read_pointer(thread_op)?)? { None } else { Some(thread_op) };

        this.start_thread(
            thread,
            start_op,
            Abi::System { unwind: false },
            arg_op,
            this.layout_of(this.tcx.types.u32)?,
        )
    }

    fn WaitForSingleObject(
        &mut self,
        handle: &OpTy<'tcx, Provenance>,
        timeout: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let thread = match Handle::from_scalar(this.read_scalar(handle)?.check_init()?, this)? {
            Some(Handle::Thread(thread)) => thread,
            // Unlike on posix, joining the current thread is not UB on windows.
            // It will just deadlock.
            Some(Handle::Pseudo(PseudoHandle::CurrentThread)) => this.get_active_thread(),
            _ => this.invalid_handle("WaitForSingleObject")?,
        };

        if this.read_scalar(timeout)?.to_u32()? != this.eval_windows("c", "INFINITE")?.to_u32()? {
            throw_unsup_format!("`WaitForSingleObject` with non-infinite timeout");
        }

        this.join_thread(thread)?;

        Ok(())
    }
}
