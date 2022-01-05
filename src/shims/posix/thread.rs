use std::convert::TryInto;

use crate::*;
use rustc_middle::ty::layout::LayoutOf;
use rustc_target::spec::abi::Abi;

impl<'mir, 'tcx> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn pthread_create(
        &mut self,
        thread: &OpTy<'tcx, Tag>,
        _attr: &OpTy<'tcx, Tag>,
        start_routine: &OpTy<'tcx, Tag>,
        arg: &OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.tcx.sess.warn(
            "thread support is experimental and incomplete: weak memory effects are not emulated.",
        );

        // Create the new thread
        let new_thread_id = this.create_thread();

        // Write the current thread-id, switch to the next thread later
        // to treat this write operation as occuring on the current thread.
        let thread_info_place = this.deref_operand(thread)?;
        this.write_scalar(
            Scalar::from_uint(new_thread_id.to_u32(), thread_info_place.layout.size),
            &thread_info_place.into(),
        )?;

        // Read the function argument that will be sent to the new thread
        // before the thread starts executing since reading after the
        // context switch will incorrectly report a data-race.
        let fn_ptr = this.read_pointer(start_routine)?;
        let func_arg = this.read_immediate(arg)?;

        // Finally switch to new thread so that we can push the first stackframe.
        // After this all accesses will be treated as occuring in the new thread.
        let old_thread_id = this.set_active_thread(new_thread_id);

        // Perform the function pointer load in the new thread frame.
        let instance = this.memory.get_fn(fn_ptr)?.as_instance()?;

        // Note: the returned value is currently ignored (see the FIXME in
        // pthread_join below) because the Rust standard library does not use
        // it.
        let ret_place =
            this.allocate(this.layout_of(this.tcx.types.usize)?, MiriMemoryKind::Machine.into())?;

        this.call_function(
            instance,
            Abi::C { unwind: false },
            &[*func_arg],
            Some(&ret_place.into()),
            StackPopCleanup::Root { cleanup: true },
        )?;

        // Restore the old active thread frame.
        this.set_active_thread(old_thread_id);

        Ok(0)
    }

    fn pthread_join(
        &mut self,
        thread: &OpTy<'tcx, Tag>,
        retval: &OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        if !this.ptr_is_null(this.read_pointer(retval)?)? {
            // FIXME: implement reading the thread function's return place.
            throw_unsup_format!("Miri supports pthread_join only with retval==NULL");
        }

        let thread_id = this.read_scalar(thread)?.to_machine_usize(this)?;
        this.join_thread(thread_id.try_into().expect("thread ID should fit in u32"))?;

        Ok(0)
    }

    fn pthread_detach(&mut self, thread: &OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let thread_id = this.read_scalar(thread)?.to_machine_usize(this)?;
        this.detach_thread(thread_id.try_into().expect("thread ID should fit in u32"))?;

        Ok(0)
    }

    fn pthread_self(&mut self, dest: &PlaceTy<'tcx, Tag>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let thread_id = this.get_active_thread();
        this.write_scalar(Scalar::from_uint(thread_id.to_u32(), dest.layout.size), dest)
    }

    fn prctl(
        &mut self,
        option: &OpTy<'tcx, Tag>,
        arg2: &OpTy<'tcx, Tag>,
        _arg3: &OpTy<'tcx, Tag>,
        _arg4: &OpTy<'tcx, Tag>,
        _arg5: &OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();
        this.assert_target_os("linux", "prctl");

        let option = this.read_scalar(option)?.to_i32()?;
        if option == this.eval_libc_i32("PR_SET_NAME")? {
            let address = this.read_pointer(arg2)?;
            let mut name = this.read_c_str(address)?.to_owned();
            // The name should be no more than 16 bytes, including the null
            // byte. Since `read_c_str` returns the string without the null
            // byte, we need to truncate to 15.
            name.truncate(15);
            this.set_active_thread_name(name);
        } else if option == this.eval_libc_i32("PR_GET_NAME")? {
            let address = this.read_pointer(arg2)?;
            let mut name = this.get_active_thread_name().to_vec();
            name.push(0u8);
            assert!(name.len() <= 16);
            this.memory.write_bytes(address, name)?;
        } else {
            throw_unsup_format!("unsupported prctl option {}", option);
        }

        Ok(0)
    }

    fn pthread_setname_np(&mut self, name: Pointer<Option<Tag>>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        this.assert_target_os("macos", "pthread_setname_np");

        let name = this.read_c_str(name)?.to_owned();
        this.set_active_thread_name(name);

        Ok(())
    }

    fn sched_yield(&mut self) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.yield_active_thread();

        Ok(0)
    }
}
