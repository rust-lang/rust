use crate::*;
use rustc_target::abi::LayoutOf;

impl<'mir, 'tcx> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn pthread_create(
        &mut self,
        thread: OpTy<'tcx, Tag>,
        _attr: OpTy<'tcx, Tag>,
        start_routine: OpTy<'tcx, Tag>,
        arg: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.tcx.sess.warn(
            "thread support is experimental. \
             For example, Miri does not detect data races yet.",
        );

        let new_thread_id = this.create_thread()?;
        // Also switch to new thread so that we can push the first stackframe.
        let old_thread_id = this.set_active_thread(new_thread_id)?;

        let thread_info_place = this.deref_operand(thread)?;
        this.write_scalar(
            Scalar::from_uint(new_thread_id.to_u128(), thread_info_place.layout.size),
            thread_info_place.into(),
        )?;

        let fn_ptr = this.read_scalar(start_routine)?.not_undef()?;
        let instance = this.memory.get_fn(fn_ptr)?.as_instance()?;

        let func_arg = this.read_immediate(arg)?;

        // Note: the returned value is currently ignored (see the FIXME in
        // pthread_join below) because the Rust standard library does not use
        // it.
        let ret_place =
            this.allocate(this.layout_of(this.tcx.types.usize)?, MiriMemoryKind::Machine.into());

        this.call_function(
            instance,
            &[*func_arg],
            Some(ret_place.into()),
            StackPopCleanup::None { cleanup: true },
        )?;

        this.set_active_thread(old_thread_id)?;

        Ok(0)
    }

    fn pthread_join(
        &mut self,
        thread: OpTy<'tcx, Tag>,
        retval: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        if !this.is_null(this.read_scalar(retval)?.not_undef()?)? {
            // FIXME: implement reading the thread function's return place.
            throw_unsup_format!("Miri supports pthread_join only with retval==NULL");
        }

        let thread_id = this.read_scalar(thread)?.to_machine_usize(this)?;
        this.join_thread(thread_id.into())?;

        Ok(0)
    }

    fn pthread_detach(&mut self, thread: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let thread_id = this.read_scalar(thread)?.to_machine_usize(this)?;
        this.detach_thread(thread_id.into())?;

        Ok(0)
    }

    fn pthread_self(&mut self, dest: PlaceTy<'tcx, Tag>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let thread_id = this.get_active_thread()?;
        this.write_scalar(Scalar::from_uint(thread_id.to_u128(), dest.layout.size), dest)
    }

    fn prctl(
        &mut self,
        option: OpTy<'tcx, Tag>,
        arg2: OpTy<'tcx, Tag>,
        _arg3: OpTy<'tcx, Tag>,
        _arg4: OpTy<'tcx, Tag>,
        _arg5: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let option = this.read_scalar(option)?.to_i32()?;
        if option == this.eval_libc_i32("PR_SET_NAME")? {
            let address = this.read_scalar(arg2)?.not_undef()?;
            let name = this.memory.read_c_str(address)?.to_owned();
            this.set_active_thread_name(name)?;
        } else if option == this.eval_libc_i32("PR_GET_NAME")? {
            let address = this.read_scalar(arg2)?.not_undef()?;
            let name = this.get_active_thread_name()?.to_vec();
            this.memory.write_bytes(address, name)?;
        } else {
            throw_unsup_format!("unsupported prctl option {}", option);
        }

        Ok(0)
    }

    fn sched_yield(&mut self) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.yield_active_thread()?;

        Ok(0)
    }
}
