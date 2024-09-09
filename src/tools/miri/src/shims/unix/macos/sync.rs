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

use crate::*;

impl<'tcx> EvalContextExtPriv<'tcx> for crate::MiriInterpCx<'tcx> {}
trait EvalContextExtPriv<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn os_unfair_lock_getid(&mut self, lock_ptr: &OpTy<'tcx>) -> InterpResult<'tcx, MutexId> {
        let this = self.eval_context_mut();
        let lock = this.deref_pointer(lock_ptr)?;
        // os_unfair_lock holds a 32-bit value, is initialized with zero and
        // must be assumed to be opaque. Therefore, we can just store our
        // internal mutex ID in the structure without anyone noticing.
        this.mutex_get_or_create_id(&lock, 0, |_| Ok(None))
    }
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn os_unfair_lock_lock(&mut self, lock_op: &OpTy<'tcx>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let id = this.os_unfair_lock_getid(lock_op)?;
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

        Ok(())
    }

    fn os_unfair_lock_trylock(
        &mut self,
        lock_op: &OpTy<'tcx>,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let id = this.os_unfair_lock_getid(lock_op)?;
        if this.mutex_is_locked(id) {
            // Contrary to the blocking lock function, this does not check for
            // reentrancy.
            this.write_scalar(Scalar::from_bool(false), dest)?;
        } else {
            this.mutex_lock(id);
            this.write_scalar(Scalar::from_bool(true), dest)?;
        }

        Ok(())
    }

    fn os_unfair_lock_unlock(&mut self, lock_op: &OpTy<'tcx>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let id = this.os_unfair_lock_getid(lock_op)?;
        if this.mutex_unlock(id)?.is_none() {
            // Matching the current macOS implementation: abort.
            throw_machine_stop!(TerminationInfo::Abort(
                "attempted to unlock an os_unfair_lock not owned by the current thread".to_owned()
            ));
        }

        Ok(())
    }

    fn os_unfair_lock_assert_owner(&mut self, lock_op: &OpTy<'tcx>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let id = this.os_unfair_lock_getid(lock_op)?;
        if !this.mutex_is_locked(id) || this.mutex_get_owner(id) != this.active_thread() {
            throw_machine_stop!(TerminationInfo::Abort(
                "called os_unfair_lock_assert_owner on an os_unfair_lock not owned by the current thread".to_owned()
            ));
        }

        Ok(())
    }

    fn os_unfair_lock_assert_not_owner(&mut self, lock_op: &OpTy<'tcx>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let id = this.os_unfair_lock_getid(lock_op)?;
        if this.mutex_is_locked(id) && this.mutex_get_owner(id) == this.active_thread() {
            throw_machine_stop!(TerminationInfo::Abort(
                "called os_unfair_lock_assert_not_owner on an os_unfair_lock owned by the current thread".to_owned()
            ));
        }

        Ok(())
    }
}
