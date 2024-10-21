use rustc_target::spec::abi::Abi;

use crate::*;

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
            Abi::C { unwind: false },
            func_arg,
            this.machine.layouts.mut_raw_ptr,
        )?;

        interp_ok(())
    }

    fn pthread_join(&mut self, thread: &OpTy<'tcx>, retval: &OpTy<'tcx>) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_mut();

        if !this.ptr_is_null(this.read_pointer(retval)?)? {
            // FIXME: implement reading the thread function's return place.
            throw_unsup_format!("Miri supports pthread_join only with retval==NULL");
        }

        let thread_id = this.read_scalar(thread)?.to_int(this.libc_ty_layout("pthread_t").size)?;
        this.join_thread_exclusive(thread_id.try_into().expect("thread ID should fit in u32"))?;

        interp_ok(())
    }

    fn pthread_detach(&mut self, thread: &OpTy<'tcx>) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_mut();

        let thread_id = this.read_scalar(thread)?.to_int(this.libc_ty_layout("pthread_t").size)?;
        this.detach_thread(
            thread_id.try_into().expect("thread ID should fit in u32"),
            /*allow_terminated_joined*/ false,
        )?;

        interp_ok(())
    }

    fn pthread_self(&mut self) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let thread_id = this.active_thread();
        interp_ok(Scalar::from_uint(thread_id.to_u32(), this.libc_ty_layout("pthread_t").size))
    }

    /// Set the name of the specified thread. If the name including the null terminator
    /// is longer than `name_max_len`, then `false` is returned.
    fn pthread_setname_np(
        &mut self,
        thread: Scalar,
        name: Scalar,
        name_max_len: usize,
    ) -> InterpResult<'tcx, bool> {
        let this = self.eval_context_mut();

        let thread = thread.to_int(this.libc_ty_layout("pthread_t").size)?;
        let thread = ThreadId::try_from(thread).unwrap();
        let name = name.to_pointer(this)?;
        let name = this.read_c_str(name)?.to_owned();

        // Comparing with `>=` to account for null terminator.
        if name.len() >= name_max_len {
            return interp_ok(false);
        }

        this.set_thread_name(thread, name);

        interp_ok(true)
    }

    /// Get the name of the specified thread. If the thread name doesn't fit
    /// the buffer, then if `truncate` is set the truncated name is written out,
    /// otherwise `false` is returned.
    fn pthread_getname_np(
        &mut self,
        thread: Scalar,
        name_out: Scalar,
        len: Scalar,
        truncate: bool,
    ) -> InterpResult<'tcx, bool> {
        let this = self.eval_context_mut();

        let thread = thread.to_int(this.libc_ty_layout("pthread_t").size)?;
        let thread = ThreadId::try_from(thread).unwrap();
        let name_out = name_out.to_pointer(this)?;
        let len = len.to_target_usize(this)?;

        // FIXME: we should use the program name if the thread name is not set
        let name = this.get_thread_name(thread).unwrap_or(b"<unnamed>").to_owned();
        let name = match truncate {
            true => &name[..name.len().min(len.try_into().unwrap_or(usize::MAX).saturating_sub(1))],
            false => &name,
        };

        let (success, _written) = this.write_c_str(name, name_out, len)?;

        interp_ok(success)
    }

    fn sched_yield(&mut self) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_mut();

        this.yield_active_thread();

        interp_ok(())
    }
}
