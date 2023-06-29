use crate::*;
use rustc_middle::ty::layout::LayoutOf;
use rustc_target::spec::abi::Abi;

impl<'mir, 'tcx> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    fn pthread_create(
        &mut self,
        thread: &OpTy<'tcx, Provenance>,
        _attr: &OpTy<'tcx, Provenance>,
        start_routine: &OpTy<'tcx, Provenance>,
        arg: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let thread_info_place = this.deref_operand_as(thread, this.libc_ty_layout("pthread_t"))?;

        let start_routine = this.read_pointer(start_routine)?;

        let func_arg = this.read_immediate(arg)?;

        this.start_regular_thread(
            Some(thread_info_place),
            start_routine,
            Abi::C { unwind: false },
            func_arg,
            this.layout_of(this.tcx.types.usize)?,
        )?;

        Ok(0)
    }

    fn pthread_join(
        &mut self,
        thread: &OpTy<'tcx, Provenance>,
        retval: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        if !this.ptr_is_null(this.read_pointer(retval)?)? {
            // FIXME: implement reading the thread function's return place.
            throw_unsup_format!("Miri supports pthread_join only with retval==NULL");
        }

        let thread_id = this.read_target_usize(thread)?;
        this.join_thread_exclusive(thread_id.try_into().expect("thread ID should fit in u32"))?;

        Ok(0)
    }

    fn pthread_detach(&mut self, thread: &OpTy<'tcx, Provenance>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let thread_id = this.read_target_usize(thread)?;
        this.detach_thread(
            thread_id.try_into().expect("thread ID should fit in u32"),
            /*allow_terminated_joined*/ false,
        )?;

        Ok(0)
    }

    fn pthread_self(&mut self) -> InterpResult<'tcx, Scalar<Provenance>> {
        let this = self.eval_context_mut();

        let thread_id = this.get_active_thread();
        Ok(Scalar::from_target_usize(thread_id.into(), this))
    }

    /// Set the name of the current thread. `max_name_len` is the maximal length of the name
    /// including the null terminator.
    fn pthread_setname_np(
        &mut self,
        thread: Scalar<Provenance>,
        name: Scalar<Provenance>,
        max_name_len: usize,
    ) -> InterpResult<'tcx, Scalar<Provenance>> {
        let this = self.eval_context_mut();

        let thread = ThreadId::try_from(thread.to_target_usize(this)?).unwrap();
        let name = name.to_pointer(this)?;

        let name = this.read_c_str(name)?.to_owned();

        // Comparing with `>=` to account for null terminator.
        if name.len() >= max_name_len {
            return Ok(this.eval_libc("ERANGE"));
        }

        this.set_thread_name(thread, name);

        Ok(Scalar::from_u32(0))
    }

    fn pthread_getname_np(
        &mut self,
        thread: Scalar<Provenance>,
        name_out: Scalar<Provenance>,
        len: Scalar<Provenance>,
    ) -> InterpResult<'tcx, Scalar<Provenance>> {
        let this = self.eval_context_mut();

        let thread = ThreadId::try_from(thread.to_target_usize(this)?).unwrap();
        let name_out = name_out.to_pointer(this)?;
        let len = len.to_target_usize(this)?;

        let name = this.get_thread_name(thread).to_owned();
        let (success, _written) = this.write_c_str(&name, name_out, len)?;

        Ok(if success { Scalar::from_u32(0) } else { this.eval_libc("ERANGE") })
    }

    fn sched_yield(&mut self) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.yield_active_thread();

        Ok(0)
    }
}
