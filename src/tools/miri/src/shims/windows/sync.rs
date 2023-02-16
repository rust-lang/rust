use std::time::Duration;

use rustc_target::abi::Size;

use crate::concurrency::init_once::InitOnceStatus;
use crate::concurrency::sync::{CondvarLock, RwLockMode};
use crate::concurrency::thread::MachineCallback;
use crate::*;

const SRWLOCK_ID_OFFSET: u64 = 0;
const INIT_ONCE_ID_OFFSET: u64 = 0;
const CONDVAR_ID_OFFSET: u64 = 0;

impl<'mir, 'tcx> EvalContextExtPriv<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
trait EvalContextExtPriv<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    /// Try to reacquire the lock associated with the condition variable after we
    /// were signaled.
    fn reacquire_cond_lock(
        &mut self,
        thread: ThreadId,
        lock: RwLockId,
        mode: RwLockMode,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        this.unblock_thread(thread);

        match mode {
            RwLockMode::Read =>
                if this.rwlock_is_write_locked(lock) {
                    this.rwlock_enqueue_and_block_reader(lock, thread);
                } else {
                    this.rwlock_reader_lock(lock, thread);
                },
            RwLockMode::Write =>
                if this.rwlock_is_locked(lock) {
                    this.rwlock_enqueue_and_block_writer(lock, thread);
                } else {
                    this.rwlock_writer_lock(lock, thread);
                },
        }

        Ok(())
    }
}

impl<'mir, 'tcx> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
#[allow(non_snake_case)]
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    fn AcquireSRWLockExclusive(&mut self, lock_op: &OpTy<'tcx, Provenance>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let id = this.rwlock_get_or_create_id(lock_op, SRWLOCK_ID_OFFSET)?;
        let active_thread = this.get_active_thread();

        if this.rwlock_is_locked(id) {
            // Note: this will deadlock if the lock is already locked by this
            // thread in any way.
            //
            // FIXME: Detect and report the deadlock proactively. (We currently
            // report the deadlock only when no thread can continue execution,
            // but we could detect that this lock is already locked and report
            // an error.)
            this.rwlock_enqueue_and_block_writer(id, active_thread);
        } else {
            this.rwlock_writer_lock(id, active_thread);
        }

        Ok(())
    }

    fn TryAcquireSRWLockExclusive(
        &mut self,
        lock_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, Scalar<Provenance>> {
        let this = self.eval_context_mut();
        let id = this.rwlock_get_or_create_id(lock_op, SRWLOCK_ID_OFFSET)?;
        let active_thread = this.get_active_thread();

        if this.rwlock_is_locked(id) {
            // Lock is already held.
            Ok(Scalar::from_u8(0))
        } else {
            this.rwlock_writer_lock(id, active_thread);
            Ok(Scalar::from_u8(1))
        }
    }

    fn ReleaseSRWLockExclusive(&mut self, lock_op: &OpTy<'tcx, Provenance>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let id = this.rwlock_get_or_create_id(lock_op, SRWLOCK_ID_OFFSET)?;
        let active_thread = this.get_active_thread();

        if !this.rwlock_writer_unlock(id, active_thread) {
            // The docs do not say anything about this case, but it seems better to not allow it.
            throw_ub_format!(
                "calling ReleaseSRWLockExclusive on an SRWLock that is not exclusively locked by the current thread"
            );
        }

        Ok(())
    }

    fn AcquireSRWLockShared(&mut self, lock_op: &OpTy<'tcx, Provenance>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let id = this.rwlock_get_or_create_id(lock_op, SRWLOCK_ID_OFFSET)?;
        let active_thread = this.get_active_thread();

        if this.rwlock_is_write_locked(id) {
            this.rwlock_enqueue_and_block_reader(id, active_thread);
        } else {
            this.rwlock_reader_lock(id, active_thread);
        }

        Ok(())
    }

    fn TryAcquireSRWLockShared(
        &mut self,
        lock_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, Scalar<Provenance>> {
        let this = self.eval_context_mut();
        let id = this.rwlock_get_or_create_id(lock_op, SRWLOCK_ID_OFFSET)?;
        let active_thread = this.get_active_thread();

        if this.rwlock_is_write_locked(id) {
            Ok(Scalar::from_u8(0))
        } else {
            this.rwlock_reader_lock(id, active_thread);
            Ok(Scalar::from_u8(1))
        }
    }

    fn ReleaseSRWLockShared(&mut self, lock_op: &OpTy<'tcx, Provenance>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let id = this.rwlock_get_or_create_id(lock_op, SRWLOCK_ID_OFFSET)?;
        let active_thread = this.get_active_thread();

        if !this.rwlock_reader_unlock(id, active_thread) {
            // The docs do not say anything about this case, but it seems better to not allow it.
            throw_ub_format!(
                "calling ReleaseSRWLockShared on an SRWLock that is not locked by the current thread"
            );
        }

        Ok(())
    }

    fn InitOnceBeginInitialize(
        &mut self,
        init_once_op: &OpTy<'tcx, Provenance>,
        flags_op: &OpTy<'tcx, Provenance>,
        pending_op: &OpTy<'tcx, Provenance>,
        context_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, Scalar<Provenance>> {
        let this = self.eval_context_mut();
        let active_thread = this.get_active_thread();

        let id = this.init_once_get_or_create_id(init_once_op, INIT_ONCE_ID_OFFSET)?;
        let flags = this.read_scalar(flags_op)?.to_u32()?;
        let pending_place = this.deref_operand(pending_op)?.into();
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
                    pending_place: PlaceTy<'tcx, Provenance>,
                }

                impl<'tcx> VisitTags for Callback<'tcx> {
                    fn visit_tags(&self, visit: &mut dyn FnMut(BorTag)) {
                        let Callback { init_once_id: _, pending_place } = self;
                        pending_place.visit_tags(visit);
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

        let id = this.init_once_get_or_create_id(init_once_op, INIT_ONCE_ID_OFFSET)?;
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
        dest: &PlaceTy<'tcx, Provenance>,
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
        let futex_val = this
            .read_scalar_atomic(&MPlaceTy::from_aligned_ptr(ptr, layout), AtomicReadOrd::Relaxed)?;
        let compare_val = this.read_scalar(&MPlaceTy::from_aligned_ptr(compare, layout).into())?;

        if futex_val == compare_val {
            // If the values are the same, we have to block.
            this.block_thread(thread);
            this.futex_wait(addr, thread, u32::MAX);

            if let Some(timeout_time) = timeout_time {
                struct Callback<'tcx> {
                    thread: ThreadId,
                    addr: u64,
                    dest: PlaceTy<'tcx, Provenance>,
                }

                impl<'tcx> VisitTags for Callback<'tcx> {
                    fn visit_tags(&self, visit: &mut dyn FnMut(BorTag)) {
                        let Callback { thread: _, addr: _, dest } = self;
                        dest.visit_tags(visit);
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

    fn SleepConditionVariableSRW(
        &mut self,
        condvar_op: &OpTy<'tcx, Provenance>,
        lock_op: &OpTy<'tcx, Provenance>,
        timeout_op: &OpTy<'tcx, Provenance>,
        flags_op: &OpTy<'tcx, Provenance>,
        dest: &PlaceTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, Scalar<Provenance>> {
        let this = self.eval_context_mut();

        let condvar_id = this.condvar_get_or_create_id(condvar_op, CONDVAR_ID_OFFSET)?;
        let lock_id = this.rwlock_get_or_create_id(lock_op, SRWLOCK_ID_OFFSET)?;
        let timeout_ms = this.read_scalar(timeout_op)?.to_u32()?;
        let flags = this.read_scalar(flags_op)?.to_u32()?;

        let timeout_time = if timeout_ms == this.eval_windows_u32("c", "INFINITE") {
            None
        } else {
            let duration = Duration::from_millis(timeout_ms.into());
            Some(this.machine.clock.now().checked_add(duration).unwrap())
        };

        let shared_mode = 0x1; // CONDITION_VARIABLE_LOCKMODE_SHARED is not in std
        let mode = if flags == 0 {
            RwLockMode::Write
        } else if flags == shared_mode {
            RwLockMode::Read
        } else {
            throw_unsup_format!("unsupported `Flags` {flags} in `SleepConditionVariableSRW`");
        };

        let active_thread = this.get_active_thread();

        let was_locked = match mode {
            RwLockMode::Read => this.rwlock_reader_unlock(lock_id, active_thread),
            RwLockMode::Write => this.rwlock_writer_unlock(lock_id, active_thread),
        };

        if !was_locked {
            throw_ub_format!(
                "calling SleepConditionVariableSRW with an SRWLock that is not locked by the current thread"
            );
        }

        this.block_thread(active_thread);
        this.condvar_wait(condvar_id, active_thread, CondvarLock::RwLock { id: lock_id, mode });

        if let Some(timeout_time) = timeout_time {
            struct Callback<'tcx> {
                thread: ThreadId,
                condvar_id: CondvarId,
                lock_id: RwLockId,
                mode: RwLockMode,
                dest: PlaceTy<'tcx, Provenance>,
            }

            impl<'tcx> VisitTags for Callback<'tcx> {
                fn visit_tags(&self, visit: &mut dyn FnMut(BorTag)) {
                    let Callback { thread: _, condvar_id: _, lock_id: _, mode: _, dest } = self;
                    dest.visit_tags(visit);
                }
            }

            impl<'mir, 'tcx: 'mir> MachineCallback<'mir, 'tcx> for Callback<'tcx> {
                fn call(&self, this: &mut MiriInterpCx<'mir, 'tcx>) -> InterpResult<'tcx> {
                    this.reacquire_cond_lock(self.thread, self.lock_id, self.mode)?;

                    this.condvar_remove_waiter(self.condvar_id, self.thread);

                    let error_timeout = this.eval_windows("c", "ERROR_TIMEOUT");
                    this.set_last_error(error_timeout)?;
                    this.write_scalar(this.eval_windows("c", "FALSE"), &self.dest)?;
                    Ok(())
                }
            }

            this.register_timeout_callback(
                active_thread,
                Time::Monotonic(timeout_time),
                Box::new(Callback {
                    thread: active_thread,
                    condvar_id,
                    lock_id,
                    mode,
                    dest: dest.clone(),
                }),
            );
        }

        Ok(this.eval_windows("c", "TRUE"))
    }

    fn WakeConditionVariable(&mut self, condvar_op: &OpTy<'tcx, Provenance>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let condvar_id = this.condvar_get_or_create_id(condvar_op, CONDVAR_ID_OFFSET)?;

        if let Some((thread, lock)) = this.condvar_signal(condvar_id) {
            if let CondvarLock::RwLock { id, mode } = lock {
                this.reacquire_cond_lock(thread, id, mode)?;
                this.unregister_timeout_callback_if_exists(thread);
            } else {
                panic!("mutexes should not exist on windows");
            }
        }

        Ok(())
    }

    fn WakeAllConditionVariable(
        &mut self,
        condvar_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let condvar_id = this.condvar_get_or_create_id(condvar_op, CONDVAR_ID_OFFSET)?;

        while let Some((thread, lock)) = this.condvar_signal(condvar_id) {
            if let CondvarLock::RwLock { id, mode } = lock {
                this.reacquire_cond_lock(thread, id, mode)?;
                this.unregister_timeout_callback_if_exists(thread);
            } else {
                panic!("mutexes should not exist on windows");
            }
        }

        Ok(())
    }
}
