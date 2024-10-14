//! Contains macOS-specific synchronization functions.
//!
//! For `os_unfair_lock`, see the documentation
//! <https://developer.apple.com/documentation/os/synchronization?language=objc>
//! and in case of underspecification its implementation
//! <https://github.com/apple-oss-distributions/libplatform/blob/a00a4cc36da2110578bcf3b8eeeeb93dcc7f4e11/src/os/lock.c#L645>.
//!
//! Note that we don't emulate every edge-case behaviour of the locks. Notably,
//! we don't abort when locking a lock owned by a thread that has already exited
//! and we do not detect copying of the lock, but macOS doesn't guarantee anything
//! in that case either.

use rustc_target::abi::Size;

use crate::*;

#[derive(Copy, Clone)]
enum MacOsUnfairLock {
    Poisoned,
    Active { id: MutexId },
}

impl<'tcx> EvalContextExtPriv<'tcx> for crate::MiriInterpCx<'tcx> {}
trait EvalContextExtPriv<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn os_unfair_lock_get_data(
        &mut self,
        lock_ptr: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, MacOsUnfairLock> {
        let this = self.eval_context_mut();
        let lock = this.deref_pointer(lock_ptr)?;
        this.lazy_sync_get_data(
            &lock,
            Size::ZERO, // offset for init tracking
            || {
                // If we get here, due to how we reset things to zero in `os_unfair_lock_unlock`,
                // this means the lock was moved while locked. This can happen with a `std` lock,
                // but then any future attempt to unlock will just deadlock. In practice, terrible
                // things can probably happen if you swap two locked locks, since they'd wake up
                // from the wrong queue... we just won't catch all UB of this library API then (we
                // would need to store some unique identifer in-memory for this, instead of a static
                // LAZY_INIT_COOKIE). This can't be hit via `std::sync::Mutex`.
                interp_ok(MacOsUnfairLock::Poisoned)
            },
            |ecx| {
                let id = ecx.machine.sync.mutex_create();
                interp_ok(MacOsUnfairLock::Active { id })
            },
        )
    }
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn os_unfair_lock_lock(&mut self, lock_op: &OpTy<'tcx>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let MacOsUnfairLock::Active { id } = this.os_unfair_lock_get_data(lock_op)? else {
            // Trying to get a poisoned lock. Just block forever...
            this.block_thread(
                BlockReason::Sleep,
                None,
                callback!(
                    @capture<'tcx> {}
                    @unblock = |_this| {
                        panic!("we shouldn't wake up ever")
                    }
                ),
            );
            return interp_ok(());
        };

        if this.mutex_is_locked(id) {
            if this.mutex_get_owner(id) == this.active_thread() {
                // Matching the current macOS implementation: abort on reentrant locking.
                throw_machine_stop!(TerminationInfo::Abort(
                    "attempted to lock an os_unfair_lock that is already locked by the current thread".to_owned()
                ));
            }

            this.mutex_enqueue_and_block(id, None);
        } else {
            this.mutex_lock(id);
        }

        interp_ok(())
    }

    fn os_unfair_lock_trylock(
        &mut self,
        lock_op: &OpTy<'tcx>,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let MacOsUnfairLock::Active { id } = this.os_unfair_lock_get_data(lock_op)? else {
            // Trying to get a poisoned lock. That never works.
            this.write_scalar(Scalar::from_bool(false), dest)?;
            return interp_ok(());
        };

        if this.mutex_is_locked(id) {
            // Contrary to the blocking lock function, this does not check for
            // reentrancy.
            this.write_scalar(Scalar::from_bool(false), dest)?;
        } else {
            this.mutex_lock(id);
            this.write_scalar(Scalar::from_bool(true), dest)?;
        }

        interp_ok(())
    }

    fn os_unfair_lock_unlock(&mut self, lock_op: &OpTy<'tcx>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let MacOsUnfairLock::Active { id } = this.os_unfair_lock_get_data(lock_op)? else {
            // The lock is poisoned, who knows who owns it... we'll pretend: someone else.
            throw_machine_stop!(TerminationInfo::Abort(
                "attempted to unlock an os_unfair_lock not owned by the current thread".to_owned()
            ));
        };

        // Now, unlock.
        if this.mutex_unlock(id)?.is_none() {
            // Matching the current macOS implementation: abort.
            throw_machine_stop!(TerminationInfo::Abort(
                "attempted to unlock an os_unfair_lock not owned by the current thread".to_owned()
            ));
        }

        // If the lock is not locked by anyone now, it went quer.
        // Reset to zero so that it can be moved and initialized again for the next phase.
        if !this.mutex_is_locked(id) {
            let lock_place = this.deref_pointer_as(lock_op, this.machine.layouts.u32)?;
            this.write_scalar_atomic(Scalar::from_u32(0), &lock_place, AtomicWriteOrd::Relaxed)?;
        }

        interp_ok(())
    }

    fn os_unfair_lock_assert_owner(&mut self, lock_op: &OpTy<'tcx>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let MacOsUnfairLock::Active { id } = this.os_unfair_lock_get_data(lock_op)? else {
            // The lock is poisoned, who knows who owns it... we'll pretend: someone else.
            throw_machine_stop!(TerminationInfo::Abort(
                "called os_unfair_lock_assert_owner on an os_unfair_lock not owned by the current thread".to_owned()
            ));
        };
        if !this.mutex_is_locked(id) || this.mutex_get_owner(id) != this.active_thread() {
            throw_machine_stop!(TerminationInfo::Abort(
                "called os_unfair_lock_assert_owner on an os_unfair_lock not owned by the current thread".to_owned()
            ));
        }

        // The lock is definitely not quiet since we are the owner.

        interp_ok(())
    }

    fn os_unfair_lock_assert_not_owner(&mut self, lock_op: &OpTy<'tcx>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let MacOsUnfairLock::Active { id } = this.os_unfair_lock_get_data(lock_op)? else {
            // The lock is poisoned, who knows who owns it... we'll pretend: someone else.
            return interp_ok(());
        };
        if this.mutex_is_locked(id) && this.mutex_get_owner(id) == this.active_thread() {
            throw_machine_stop!(TerminationInfo::Abort(
                "called os_unfair_lock_assert_not_owner on an os_unfair_lock owned by the current thread".to_owned()
            ));
        }

        // If the lock is not locked by anyone now, it went quer.
        // Reset to zero so that it can be moved and initialized again for the next phase.
        if !this.mutex_is_locked(id) {
            let lock_place = this.deref_pointer_as(lock_op, this.machine.layouts.u32)?;
            this.write_scalar_atomic(Scalar::from_u32(0), &lock_place, AtomicWriteOrd::Relaxed)?;
        }

        interp_ok(())
    }
}
