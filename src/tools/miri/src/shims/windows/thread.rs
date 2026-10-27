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
        let ecx = self.eval_context_mut();

        let security = ecx.read_pointer(security_op)?;
        // stacksize is ignored, but still needs to be a valid usize
        ecx.read_target_usize(stacksize_op)?;
        let start_routine = ecx.read_pointer(start_op)?;
        let func_arg = ecx.read_immediate(arg_op)?;
        let flags = ecx.read_scalar(flags_op)?.to_u32()?;

        let thread = if ecx.ptr_is_null(ecx.read_pointer(thread_op)?)? {
            None
        } else {
            let thread_info_place = ecx.deref_pointer_as(thread_op, ecx.machine.layouts.u32)?;
            Some(thread_info_place)
        };

        let stack_size_param_is_a_reservation =
            ecx.eval_windows_u32("c", "STACK_SIZE_PARAM_IS_A_RESERVATION");

        // We ignore the stack size, so we also ignore the
        // `STACK_SIZE_PARAM_IS_A_RESERVATION` flag.
        if flags != 0 && flags != stack_size_param_is_a_reservation {
            throw_unsup_format!("unsupported `dwCreationFlags` {} in `CreateThread`", flags)
        }

        if !ecx.ptr_is_null(security)? {
            throw_unsup_format!("non-null `lpThreadAttributes` in `CreateThread`")
        }

        ecx.start_regular_thread(
            thread,
            start_routine,
            ExternAbi::System { unwind: false },
            func_arg,
            ecx.layout_of(ecx.tcx.types.u32)?,
        )
    }

    fn WaitForSingleObject(
        &mut self,
        handle_op: &OpTy<'tcx>,
        timeout_op: &OpTy<'tcx>,
        return_dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let ecx = self.eval_context_mut();

        let handle = ecx.read_handle(handle_op, "WaitForSingleObject")?;
        let timeout = ecx.read_scalar(timeout_op)?.to_u32()?;

        let joined_thread_id = match handle {
            Handle::Thread(thread) => thread,
            // Unlike on posix, the outcome of joining the current thread is not documented.
            // On current Windows, it just deadlocks.
            Handle::Pseudo(PseudoHandle::CurrentThread) => ecx.active_thread(),
            _ => ecx.invalid_handle("WaitForSingleObject")?,
        };

        if timeout != ecx.eval_windows_u32("c", "INFINITE") {
            throw_unsup_format!("`WaitForSingleObject` with non-infinite timeout");
        }

        ecx.join_thread(
            joined_thread_id,
            /* success_retval */ ecx.eval_windows("c", "WAIT_OBJECT_0"),
            return_dest,
        )?;

        interp_ok(())
    }
}
