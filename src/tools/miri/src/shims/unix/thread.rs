use rustc_abi::ExternAbi;

use crate::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThreadNameResult {
    Ok,
    NameTooLong,
    ThreadNotFound,
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn pthread_create(
        &mut self,
        thread: &OpTy<'tcx>,
        _attr: &OpTy<'tcx>,
        start_routine: &OpTy<'tcx>,
        arg: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_mut();

        let thread_info_place = this.deref_pointer_as(thread, this.libc_ty_layout("pthread_t"))?;

        let start_routine = this.read_pointer(start_routine)?;

        let func_arg = this.read_immediate(arg)?;

        this.start_regular_thread(
            Some(thread_info_place),
            start_routine,
            ExternAbi::C { unwind: false },
            func_arg,
            this.machine.layouts.mut_raw_ptr,
        )?;

        interp_ok(())
    }

    fn pthread_join(
        &mut self,
        thread: &OpTy<'tcx>,
        retval: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        if !this.ptr_is_null(this.read_pointer(retval)?)? {
            // FIXME: implement reading the thread function's return place.
            throw_unsup_format!("Miri supports pthread_join only with retval==NULL");
        }

        let thread = this.read_scalar(thread)?.to_int(this.libc_ty_layout("pthread_t").size)?;
        let Ok(thread) = this.thread_id_try_from(thread) else {
            return interp_ok(this.eval_libc("ESRCH"));
        };

        this.join_thread_exclusive(thread)?;

        interp_ok(Scalar::from_u32(0))
    }

    fn pthread_detach(&mut self, thread: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let thread = this.read_scalar(thread)?.to_int(this.libc_ty_layout("pthread_t").size)?;
        let Ok(thread) = this.thread_id_try_from(thread) else {
            return interp_ok(this.eval_libc("ESRCH"));
        };
        this.detach_thread(thread, /*allow_terminated_joined*/ false)?;

        interp_ok(Scalar::from_u32(0))
    }

    fn pthread_self(&mut self) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let thread_id = this.active_thread();
        interp_ok(Scalar::from_uint(thread_id.to_u32(), this.libc_ty_layout("pthread_t").size))
    }

    /// Set the name of the specified thread. If the name including the null terminator
    /// is longer or equals to `name_max_len`, then if `truncate` is set the truncated name
    /// is used as the thread name, otherwise [`ThreadNameResult::NameTooLong`] is returned.
    /// If the specified thread wasn't found, [`ThreadNameResult::ThreadNotFound`] is returned.
    fn pthread_setname_np(
        &mut self,
        thread: Scalar,
        name: Scalar,
        name_max_len: u64,
        truncate: bool,
    ) -> InterpResult<'tcx, ThreadNameResult> {
        let this = self.eval_context_mut();

        let thread = thread.to_int(this.libc_ty_layout("pthread_t").size)?;
        let Ok(thread) = this.thread_id_try_from(thread) else {
            return interp_ok(ThreadNameResult::ThreadNotFound);
        };
        let name = name.to_pointer(this)?;
        let mut name = this.read_c_str(name)?.to_owned();

        // Comparing with `>=` to account for null terminator.
        if name.len().to_u64() >= name_max_len {
            if truncate {
                name.truncate(name_max_len.saturating_sub(1).try_into().unwrap());
            } else {
                return interp_ok(ThreadNameResult::NameTooLong);
            }
        }

        this.set_thread_name(thread, name);

        interp_ok(ThreadNameResult::Ok)
    }

    /// Get the name of the specified thread. If the thread name doesn't fit
    /// the buffer, then if `truncate` is set the truncated name is written out,
    /// otherwise [`ThreadNameResult::NameTooLong`] is returned. If the specified
    /// thread wasn't found, [`ThreadNameResult::ThreadNotFound`] is returned.
    fn pthread_getname_np(
        &mut self,
        thread: Scalar,
        name_out: Scalar,
        len: Scalar,
        truncate: bool,
    ) -> InterpResult<'tcx, ThreadNameResult> {
        let this = self.eval_context_mut();

        let thread = thread.to_int(this.libc_ty_layout("pthread_t").size)?;
        let Ok(thread) = this.thread_id_try_from(thread) else {
            return interp_ok(ThreadNameResult::ThreadNotFound);
        };
        let name_out = name_out.to_pointer(this)?;
        let len = len.to_target_usize(this)?;

        // FIXME: we should use the program name if the thread name is not set
        let name = this.get_thread_name(thread).unwrap_or(b"<unnamed>").to_owned();
        let name = match truncate {
            true => &name[..name.len().min(len.try_into().unwrap_or(usize::MAX).saturating_sub(1))],
            false => &name,
        };

        let (success, _written) = this.write_c_str(name, name_out, len)?;
        let res = if success { ThreadNameResult::Ok } else { ThreadNameResult::NameTooLong };

        interp_ok(res)
    }

    fn sched_yield(&mut self) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_mut();

        this.yield_active_thread();

        interp_ok(())
    }
}
