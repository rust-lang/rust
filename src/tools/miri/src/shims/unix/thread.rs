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

        let thread_info_place = this.deref_operand(thread)?;

        let start_routine = this.read_pointer(start_routine)?;

        let func_arg = this.read_immediate(arg)?;

        this.start_thread(
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

        let thread_id = this.read_scalar(thread)?.to_machine_usize(this)?;
        this.join_thread_exclusive(thread_id.try_into().expect("thread ID should fit in u32"))?;

        Ok(0)
    }

    fn pthread_detach(&mut self, thread: &OpTy<'tcx, Provenance>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let thread_id = this.read_scalar(thread)?.to_machine_usize(this)?;
        this.detach_thread(
            thread_id.try_into().expect("thread ID should fit in u32"),
            /*allow_terminated_joined*/ false,
        )?;

        Ok(0)
    }

    fn pthread_self(&mut self) -> InterpResult<'tcx, Scalar<Provenance>> {
        let this = self.eval_context_mut();

        let thread_id = this.get_active_thread();
        Ok(Scalar::from_machine_usize(thread_id.into(), this))
    }

    fn pthread_setname_np(
        &mut self,
        thread: Scalar<Provenance>,
        name: Scalar<Provenance>,
    ) -> InterpResult<'tcx, Scalar<Provenance>> {
        let this = self.eval_context_mut();

        let thread = ThreadId::try_from(thread.to_machine_usize(this)?).unwrap();
        let name = name.to_pointer(this)?;

        let name = this.read_c_str(name)?.to_owned();
        this.set_thread_name(thread, name);

        Ok(Scalar::from_u32(0))
    }

    fn sched_yield(&mut self) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.yield_active_thread();

        Ok(0)
    }
}
