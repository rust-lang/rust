use std::time::Duration;

use rustc_target::abi::Size;

use crate::concurrency::init_once::InitOnceStatus;
use crate::concurrency::thread::MachineCallback;
use crate::*;

impl<'mir, 'tcx> EvalContextExtPriv<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
trait EvalContextExtPriv<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    // Windows sync primitives are pointer sized.
    // We only use the first 4 bytes for the id.

    fn init_once_get_id(
        &mut self,
        init_once_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, InitOnceId> {
        let this = self.eval_context_mut();
        this.init_once_get_or_create_id(init_once_op, this.windows_ty_layout("INIT_ONCE"), 0)
    }
}

impl<'mir, 'tcx> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
#[allow(non_snake_case)]
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    fn InitOnceBeginInitialize(
        &mut self,
        init_once_op: &OpTy<'tcx, Provenance>,
        flags_op: &OpTy<'tcx, Provenance>,
        pending_op: &OpTy<'tcx, Provenance>,
        context_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, Scalar<Provenance>> {
        let this = self.eval_context_mut();
        let active_thread = this.get_active_thread();

        let id = this.init_once_get_id(init_once_op)?;
        let flags = this.read_scalar(flags_op)?.to_u32()?;
        let pending_place = this.deref_pointer(pending_op)?;
        let context = this.read_pointer(context_op)?;

        if flags != 0 {
            throw_unsup_format!("unsupported `dwFlags` {flags} in `InitOnceBeginInitialize`");
        }

        if !this.ptr_is_null(context)? {
            throw_unsup_format!("non-null `lpContext` in `InitOnceBeginInitialize`");
        }

        match this.init_once_status(id) {
            InitOnceStatus::Uninitialized => {
                this.init_once_begin(id);
                this.write_scalar(this.eval_windows("c", "TRUE"), &pending_place)?;
            }
            InitOnceStatus::Begun => {
                // Someone else is already on it.
                // Block this thread until they are done.
                // When we are woken up, set the `pending` flag accordingly.
                struct Callback<'tcx> {
                    init_once_id: InitOnceId,
                    pending_place: MPlaceTy<'tcx, Provenance>,
                }

                impl<'tcx> VisitProvenance for Callback<'tcx> {
                    fn visit_provenance(&self, visit: &mut VisitWith<'_>) {
                        let Callback { init_once_id: _, pending_place } = self;
                        pending_place.visit_provenance(visit);
                    }
                }

                impl<'mir, 'tcx> MachineCallback<'mir, 'tcx> for Callback<'tcx> {
                    fn call(&self, this: &mut MiriInterpCx<'mir, 'tcx>) -> InterpResult<'tcx> {
                        let pending = match this.init_once_status(self.init_once_id) {
                            InitOnceStatus::Uninitialized =>
                                unreachable!(
                                    "status should have either been set to begun or complete"
                                ),
                            InitOnceStatus::Begun => this.eval_windows("c", "TRUE"),
                            InitOnceStatus::Complete => this.eval_windows("c", "FALSE"),
                        };

                        this.write_scalar(pending, &self.pending_place)?;

                        Ok(())
                    }
                }

                this.init_once_enqueue_and_block(
                    id,
                    active_thread,
                    Box::new(Callback { init_once_id: id, pending_place }),
                )
            }
            InitOnceStatus::Complete => {
                this.init_once_observe_completed(id);
                this.write_scalar(this.eval_windows("c", "FALSE"), &pending_place)?;
            }
        }

        // This always succeeds (even if the thread is blocked, we will succeed if we ever unblock).
        Ok(this.eval_windows("c", "TRUE"))
    }

    fn InitOnceComplete(
        &mut self,
        init_once_op: &OpTy<'tcx, Provenance>,
        flags_op: &OpTy<'tcx, Provenance>,
        context_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, Scalar<Provenance>> {
        let this = self.eval_context_mut();

        let id = this.init_once_get_id(init_once_op)?;
        let flags = this.read_scalar(flags_op)?.to_u32()?;
        let context = this.read_pointer(context_op)?;

        let success = if flags == 0 {
            true
        } else if flags == this.eval_windows_u32("c", "INIT_ONCE_INIT_FAILED") {
            false
        } else {
            throw_unsup_format!("unsupported `dwFlags` {flags} in `InitOnceBeginInitialize`");
        };

        if !this.ptr_is_null(context)? {
            throw_unsup_format!("non-null `lpContext` in `InitOnceBeginInitialize`");
        }

        if this.init_once_status(id) != InitOnceStatus::Begun {
            // The docs do not say anything about this case, but it seems better to not allow it.
            throw_ub_format!(
                "calling InitOnceComplete on a one time initialization that has not begun or is already completed"
            );
        }

        if success {
            this.init_once_complete(id)?;
        } else {
            this.init_once_fail(id)?;
        }

        Ok(this.eval_windows("c", "TRUE"))
    }

    fn WaitOnAddress(
        &mut self,
        ptr_op: &OpTy<'tcx, Provenance>,
        compare_op: &OpTy<'tcx, Provenance>,
        size_op: &OpTy<'tcx, Provenance>,
        timeout_op: &OpTy<'tcx, Provenance>,
        dest: &MPlaceTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let ptr = this.read_pointer(ptr_op)?;
        let compare = this.read_pointer(compare_op)?;
        let size = this.read_target_usize(size_op)?;
        let timeout_ms = this.read_scalar(timeout_op)?.to_u32()?;

        let thread = this.get_active_thread();
        let addr = ptr.addr().bytes();

        if size > 8 || !size.is_power_of_two() {
            let invalid_param = this.eval_windows("c", "ERROR_INVALID_PARAMETER");
            this.set_last_error(invalid_param)?;
            this.write_scalar(Scalar::from_i32(0), dest)?;
            return Ok(());
        };
        let size = Size::from_bytes(size);

        let timeout_time = if timeout_ms == this.eval_windows_u32("c", "INFINITE") {
            None
        } else {
            let duration = Duration::from_millis(timeout_ms.into());
            Some(Time::Monotonic(this.machine.clock.now().checked_add(duration).unwrap()))
        };

        // See the Linux futex implementation for why this fence exists.
        this.atomic_fence(AtomicFenceOrd::SeqCst)?;

        let layout = this.machine.layouts.uint(size).unwrap();
        let futex_val =
            this.read_scalar_atomic(&this.ptr_to_mplace(ptr, layout), AtomicReadOrd::Relaxed)?;
        let compare_val = this.read_scalar(&this.ptr_to_mplace(compare, layout))?;

        if futex_val == compare_val {
            // If the values are the same, we have to block.
            this.block_thread(thread);
            this.futex_wait(addr, thread, u32::MAX);

            if let Some(timeout_time) = timeout_time {
                struct Callback<'tcx> {
                    thread: ThreadId,
                    addr: u64,
                    dest: MPlaceTy<'tcx, Provenance>,
                }

                impl<'tcx> VisitProvenance for Callback<'tcx> {
                    fn visit_provenance(&self, visit: &mut VisitWith<'_>) {
                        let Callback { thread: _, addr: _, dest } = self;
                        dest.visit_provenance(visit);
                    }
                }

                impl<'mir, 'tcx: 'mir> MachineCallback<'mir, 'tcx> for Callback<'tcx> {
                    fn call(&self, this: &mut MiriInterpCx<'mir, 'tcx>) -> InterpResult<'tcx> {
                        this.unblock_thread(self.thread);
                        this.futex_remove_waiter(self.addr, self.thread);
                        let error_timeout = this.eval_windows("c", "ERROR_TIMEOUT");
                        this.set_last_error(error_timeout)?;
                        this.write_scalar(Scalar::from_i32(0), &self.dest)?;

                        Ok(())
                    }
                }

                this.register_timeout_callback(
                    thread,
                    timeout_time,
                    Box::new(Callback { thread, addr, dest: dest.clone() }),
                );
            }
        }

        this.write_scalar(Scalar::from_i32(1), dest)?;

        Ok(())
    }

    fn WakeByAddressSingle(&mut self, ptr_op: &OpTy<'tcx, Provenance>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let ptr = this.read_pointer(ptr_op)?;

        // See the Linux futex implementation for why this fence exists.
        this.atomic_fence(AtomicFenceOrd::SeqCst)?;

        if let Some(thread) = this.futex_wake(ptr.addr().bytes(), u32::MAX) {
            this.unblock_thread(thread);
            this.unregister_timeout_callback_if_exists(thread);
        }

        Ok(())
    }
    fn WakeByAddressAll(&mut self, ptr_op: &OpTy<'tcx, Provenance>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let ptr = this.read_pointer(ptr_op)?;

        // See the Linux futex implementation for why this fence exists.
        this.atomic_fence(AtomicFenceOrd::SeqCst)?;

        while let Some(thread) = this.futex_wake(ptr.addr().bytes(), u32::MAX) {
            this.unblock_thread(thread);
            this.unregister_timeout_callback_if_exists(thread);
        }

        Ok(())
    }
}
