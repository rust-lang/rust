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
        let this = self.eval_context_mut();

        this.tcx.sess.warn(
            "thread support is experimental. \
             For example, Miri does not detect data races yet.",
        );

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

        let func_arg = this.read_immediate(arg)?;
        let func_args = [*func_arg];

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

    fn pthread_self(&mut self, dest: PlaceTy<'tcx, Tag>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let thread_id = this.get_active_thread()?;
        this.write_scalar(Scalar::from_uint(thread_id.index() as u128, dest.layout.size), dest)
    }

    fn prctl(
        &mut self,
        option: OpTy<'tcx, Tag>,
        arg2: OpTy<'tcx, Tag>,
        arg3: OpTy<'tcx, Tag>,
        arg4: OpTy<'tcx, Tag>,
        arg5: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        // prctl last 5 arguments are declared as variadic. Therefore, we need
        // to check their types manually.
        let c_long_size = this.libc_ty_layout("c_long")?.size.bytes();
        let check_arg = |arg: OpTy<'tcx, Tag>| -> InterpResult<'tcx> {
            match this.read_scalar(arg)?.not_undef()? {
                Scalar::Raw { size, .. } if u64::from(size) == c_long_size => Ok(()),
                _ => throw_ub_format!("an argument of unsupported type was passed to prctl"),
            }
        };
        check_arg(arg2)?;
        check_arg(arg3)?;
        check_arg(arg4)?;
        check_arg(arg5)?;

        let option = this.read_scalar(option)?.not_undef()?.to_i32()?;
        if option == this.eval_libc_i32("PR_SET_NAME")? {
            let address = this.read_scalar(arg2)?.not_undef()?;
            let name = this.memory.read_c_str(address)?.to_owned();
            this.set_active_thread_name(name)?;
        } else if option == this.eval_libc_i32("PR_GET_NAME")? {
            let address = this.read_scalar(arg2)?.not_undef()?;
            let name = this.get_active_thread_name()?;
            this.memory.write_bytes(address, name)?;
        } else {
            throw_unsup_format!("Unsupported prctl option.");
        }

        Ok(0)
    }

    fn sched_yield(&mut self) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.yield_active_thread()?;

        Ok(0)
    }
}
