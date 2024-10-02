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

    /// Set the name of the current thread. `max_name_len` is the maximal length of the name
    /// including the null terminator.
    fn pthread_setname_np(
        &mut self,
        thread: Scalar,
        name: Scalar,
        max_name_len: usize,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let thread = thread.to_int(this.libc_ty_layout("pthread_t").size)?;
        let thread = ThreadId::try_from(thread).unwrap();
        let name = name.to_pointer(this)?;

        let name = this.read_c_str(name)?.to_owned();

        // Comparing with `>=` to account for null terminator.
        if name.len() >= max_name_len {
            return interp_ok(this.eval_libc("ERANGE"));
        }

        this.set_thread_name(thread, name);

        interp_ok(Scalar::from_u32(0))
    }

    fn pthread_getname_np(
        &mut self,
        thread: Scalar,
        name_out: Scalar,
        len: Scalar,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let thread = thread.to_int(this.libc_ty_layout("pthread_t").size)?;
        let thread = ThreadId::try_from(thread).unwrap();
        let name_out = name_out.to_pointer(this)?;
        let len = len.to_target_usize(this)?;

        // FIXME: we should use the program name if the thread name is not set
        let name = this.get_thread_name(thread).unwrap_or(b"<unnamed>").to_owned();
        let (success, _written) = this.write_c_str(&name, name_out, len)?;

        interp_ok(if success { Scalar::from_u32(0) } else { this.eval_libc("ERANGE") })
    }

    fn sched_yield(&mut self) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_mut();

        this.yield_active_thread();

        interp_ok(())
    }
}
