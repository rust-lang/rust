use rustc_abi::ExternAbi;
use rustc_target::spec::Os;

use crate::concurrency::thread::ThreadLookupError;
use crate::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ThreadNameResult {
    Ok,
    NameTooLong,
}

// # pthread_attr_t

struct PthreadAttr {
    /// `None` represents that the stack size has not been explicitly set.
    stack_size: Option<u64>,
}

impl MetaObj for PthreadAttr {}

/// Returns the thread attribute data stored at the address that `attr_ptr` points to.
fn attr_get_data<'tcx, 'a>(
    ecx: &'a mut MiriInterpCx<'tcx>,
    attr_ptr: &OpTy<'tcx>,
) -> InterpResult<'tcx, &'a mut PthreadAttr>
where
    'tcx: 'a,
{
    let attr = ecx.deref_pointer(attr_ptr)?;
    ecx.get_or_insert_meta(&attr, |_ecx| {
        throw_ub_format!("`pthread_attr_t` has not been initialized or has been tampered with")
    })
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn pthread_attr_init(
        &mut self,
        attr_op: &OpTy<'tcx>,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let ecx = self.eval_context_mut();
        let attr = ecx.deref_pointer(attr_op)?;
        ecx.insert_meta(&attr, Box::new(PthreadAttr { stack_size: None }))?;
        ecx.write_int(0, dest)
    }

    fn pthread_attr_destroy(
        &mut self,
        attr_op: &OpTy<'tcx>,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let ecx = self.eval_context_mut();
        let attr = ecx.deref_pointer(attr_op)?;
        ecx.remove_meta::<PthreadAttr>(&attr)?.ok_or_else(|| {
            err_ub_format!("`pthread_attr_t` has not been initialized or has been tampered with")
        })?;
        ecx.write_int(0, dest)
    }

    fn pthread_attr_setstacksize(
        &mut self,
        attr_op: &OpTy<'tcx>,
        stack_size_op: &OpTy<'tcx>,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let ecx = self.eval_context_mut();
        let stack_size = ecx.read_target_usize(stack_size_op)?;

        // Check the attribute object even if the size is invalid.
        attr_get_data(ecx, attr_op)?;

        // Check that the size is valid.
        let os = &ecx.tcx.sess.target.os;
        // NetBSD has no `PTHREAD_STACK_MIN` constant.
        if *os != Os::NetBsd
            && stack_size < ecx.eval_libc("PTHREAD_STACK_MIN").to_target_usize(ecx)?
        {
            return ecx.write_scalar(ecx.eval_libc("EINVAL"), dest);
        }
        // macOS additionally requires the size to be a multiple of the page size.
        if *os == Os::MacOs && !stack_size.is_multiple_of(ecx.machine.page_size) {
            return ecx.write_scalar(ecx.eval_libc("EINVAL"), dest);
        }

        // Update size in `attr`.
        attr_get_data(ecx, attr_op)?.stack_size = Some(stack_size);
        ecx.write_scalar(Scalar::from_i32(0), dest)
    }

    fn pthread_create(
        &mut self,
        thread_id: &OpTy<'tcx>,
        attr: &OpTy<'tcx>,
        start_routine: &OpTy<'tcx>,
        arg: &OpTy<'tcx>,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let ecx = self.eval_context_mut();

        let thread_id_place = ecx.deref_pointer(thread_id)?;

        let start_routine = ecx.read_pointer(start_routine)?;

        let func_arg = ecx.read_immediate(arg)?;

        // A null `attr` means default attributes.
        let stack_size = if ecx.ptr_is_null(ecx.read_pointer(attr)?)? {
            None
        } else {
            attr_get_data(ecx, attr)?.stack_size
        };
        if let Some(stack_size) = stack_size
            && !ecx.thread_stack_size_ok(stack_size)
        {
            // There are not enough resources for such a big stack.
            return ecx.write_scalar(ecx.eval_libc("EAGAIN"), dest);
        }

        let thread_id = ecx.start_regular_thread(
            start_routine,
            ExternAbi::C { unwind: false },
            func_arg,
            ecx.machine.layouts.unit_ptr_mut,
        )?;
        ecx.write_int(thread_id.to_u32(), &thread_id_place)?; // pthread_t is just the Miri ThreadId
        ecx.write_scalar(Scalar::from_i32(0), dest)
    }

    fn pthread_join(
        &mut self,
        thread: &OpTy<'tcx>,
        retval: &OpTy<'tcx>,
        return_dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let ecx = self.eval_context_mut();

        if !ecx.ptr_is_null(ecx.read_pointer(retval)?)? {
            // FIXME: implement reading the thread function's return place.
            throw_unsup_format!("Miri supports pthread_join only with retval==NULL");
        }

        let thread = ecx.read_int::<i128>(thread)?;
        // Joining a terminated thread is valid.
        let thread = match ecx.thread_id_try_from(thread) {
            Ok(id) | Err(ThreadLookupError::Terminated(id)) => id,
            Err(ThreadLookupError::InvalidId) =>
                throw_ub_format!("pthread_join: invalid pthread_t handle"),
        };

        ecx.join_thread_exclusive(
            thread,
            /* success_retval */ Scalar::from_u32(0),
            return_dest,
        )
    }

    fn pthread_detach(&mut self, thread: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let ecx = self.eval_context_mut();

        let thread = ecx.read_int::<i128>(thread)?;
        // Detaching a terminated thread is valid.
        let thread = match ecx.thread_id_try_from(thread) {
            Ok(id) | Err(ThreadLookupError::Terminated(id)) => id,
            Err(ThreadLookupError::InvalidId) =>
                throw_ub_format!("pthread_detach: invalid pthread_t handle"),
        };
        ecx.detach_thread(thread, /*allow_terminated_joined*/ false)?;

        interp_ok(Scalar::from_u32(0))
    }

    fn pthread_self(&mut self, dest: &MPlaceTy<'tcx>) -> InterpResult<'tcx> {
        let ecx = self.eval_context_mut();

        let thread_id = ecx.active_thread();
        ecx.write_int(thread_id.to_u32(), dest) // pthread_t is just the Miri ThreadId
    }

    /// Set the name of the specified thread. `None` indicates the current thread.
    /// If the name including the null terminator
    /// is longer or equals to `name_max_len`, then if `truncate` is set the truncated name
    /// is used as the thread name, otherwise [`ThreadNameResult::NameTooLong`] is returned.
    /// If the specified thread wasn't found, UB is raised.
    fn pthread_setname_np(
        &mut self,
        thread: Option<&OpTy<'tcx>>,
        name: Pointer,
        name_max_len: u64,
        truncate: bool,
    ) -> InterpResult<'tcx, ThreadNameResult> {
        let ecx = self.eval_context_mut();

        let thread = match thread {
            None => ecx.active_thread(),
            Some(thread) => {
                let thread = ecx.read_int::<i128>(thread)?;
                let Ok(thread) = ecx.thread_id_try_from(thread) else {
                    // On Linux this has been observed to segfault. FreeBSD documents an error code
                    // for this condition but it is not clear whether they *guarantee* that error.
                    throw_ub_format!("pthread_setname_np: invalid pthread_t handle");
                };
                thread
            }
        };
        let mut name = ecx.read_c_str(name)?.to_owned();

        // Comparing with `>=` to account for null terminator.
        if name.len().to_u64() >= name_max_len {
            if truncate {
                name.truncate(name_max_len.saturating_sub(1).try_into().unwrap());
            } else {
                return interp_ok(ThreadNameResult::NameTooLong);
            }
        }

        ecx.set_thread_name(thread, name);

        interp_ok(ThreadNameResult::Ok)
    }

    /// Get the name of the specified thread. If the thread name doesn't fit
    /// the buffer, then if `truncate` is set the truncated name is written out,
    /// otherwise [`ThreadNameResult::NameTooLong`] is returned. If the specified
    /// thread wasn't found, UB is raised.
    fn pthread_getname_np(
        &mut self,
        thread: Option<&OpTy<'tcx>>,
        name_out: Pointer,
        len: Scalar,
        truncate: bool,
    ) -> InterpResult<'tcx, ThreadNameResult> {
        let ecx = self.eval_context_mut();

        let thread = match thread {
            None => ecx.active_thread(),
            Some(thread) => {
                let thread = ecx.read_int::<i128>(thread)?;
                let Ok(thread) = ecx.thread_id_try_from(thread) else {
                    // On Linux this has been observed to segfault. FreeBSD documents an error code
                    // for this condition but it is not clear whether they *guarantee* that error.
                    throw_ub_format!("pthread_getname_np: invalid pthread_t handle");
                };
                thread
            }
        };
        let len = len.to_target_usize(ecx)?;

        // FIXME: we should use the program name if the thread name is not set
        let name = ecx.get_thread_name(thread).unwrap_or(b"<unnamed>").to_owned();
        let name = match truncate {
            true => &name[..name.len().min(len.try_into().unwrap_or(usize::MAX).saturating_sub(1))],
            false => &name,
        };

        let (success, _written) = ecx.write_c_str(name, name_out, len)?;
        let res = if success { ThreadNameResult::Ok } else { ThreadNameResult::NameTooLong };

        interp_ok(res)
    }

    fn sched_yield(&mut self) -> InterpResult<'tcx, ()> {
        let ecx = self.eval_context_mut();

        ecx.yield_active_thread();

        interp_ok(())
    }
}
