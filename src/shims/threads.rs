use crate::*;
use rustc_index::vec::Idx;
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
        println!(
            "WARNING: The thread support is experimental. \
                  For example, Miri does not detect data races yet."
        );

        let this = self.eval_context_mut();

        let new_thread_id = this.create_thread()?;
        let old_thread_id = this.set_active_thread(new_thread_id)?;

        let thread_info_place = this.deref_operand(thread)?;
        let thread_info_type = thread.layout.ty
            .builtin_deref(true)
            .ok_or_else(|| err_ub_format!(
                "wrong signature used for `pthread_create`: first argument must be a raw pointer."
            ))?
            .ty;
        let thread_info_layout = this.layout_of(thread_info_type)?;
        this.write_scalar(
            Scalar::from_uint(new_thread_id.index() as u128, thread_info_layout.size),
            thread_info_place.into(),
        )?;

        let fn_ptr = this.read_scalar(start_routine)?.not_undef()?;
        let instance = this.memory.get_fn(fn_ptr)?.as_instance()?;

        let func_arg = match *arg {
            rustc_mir::interpret::Operand::Immediate(immediate) => immediate,
            _ => unreachable!(),
        };
        let func_args = [func_arg];

        let ret_place =
            this.allocate(this.layout_of(this.tcx.types.usize)?, MiriMemoryKind::Machine.into());

        this.call_function(
            instance,
            &func_args[..],
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
            throw_unsup_format!("Miri supports pthread_join only with retval==NULL");
        }

        let thread_id = this.read_scalar(thread)?.not_undef()?.to_machine_usize(this)?;
        this.join_thread(thread_id.into())?;

        Ok(0)
    }
    fn pthread_detach(&mut self, thread: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let thread_id = this.read_scalar(thread)?.not_undef()?.to_machine_usize(this)?;
        this.detach_thread(thread_id.into())?;

        Ok(0)
    }
}
