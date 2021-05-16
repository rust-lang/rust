use crate::*;

// Locks are pointer-sized pieces of data, initialized to 0.
// We use the first 4 bytes to store the RwLockId.

fn srwlock_get_or_create_id<'mir, 'tcx: 'mir>(
    ecx: &mut MiriEvalContext<'mir, 'tcx>,
    lock_op: &OpTy<'tcx, Tag>,
) -> InterpResult<'tcx, RwLockId> {
    let id = ecx.read_scalar_at_offset(lock_op, 0, ecx.machine.layouts.u32)?.to_u32()?;
    if id == 0 {
        // 0 is a default value and also not a valid rwlock id. Need to allocate
        // a new rwlock.
        let id = ecx.rwlock_create();
        ecx.write_scalar_at_offset(lock_op, 0, id.to_u32_scalar(), ecx.machine.layouts.u32)?;
        Ok(id)
    } else {
        Ok(RwLockId::from_u32(id))
    }
}

impl<'mir, 'tcx> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    #[allow(non_snake_case)]
    fn AcquireSRWLockExclusive(&mut self, lock_op: &OpTy<'tcx, Tag>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let id = srwlock_get_or_create_id(this, lock_op)?;
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

    #[allow(non_snake_case)]
    fn TryAcquireSRWLockExclusive(&mut self, lock_op: &OpTy<'tcx, Tag>) -> InterpResult<'tcx, u8> {
        let this = self.eval_context_mut();
        let id = srwlock_get_or_create_id(this, lock_op)?;
        let active_thread = this.get_active_thread();

        if this.rwlock_is_locked(id) {
            // Lock is already held.
            Ok(0)
        } else {
            this.rwlock_writer_lock(id, active_thread);
            Ok(1)
        }
    }

    #[allow(non_snake_case)]
    fn ReleaseSRWLockExclusive(&mut self, lock_op: &OpTy<'tcx, Tag>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let id = srwlock_get_or_create_id(this, lock_op)?;
        let active_thread = this.get_active_thread();

        if !this.rwlock_writer_unlock(id, active_thread) {
            // The docs do not say anything about this case, but it seems better to not allow it.
            throw_ub_format!(
                "calling ReleaseSRWLockExclusive on an SRWLock that is not exclusively locked by the current thread"
            );
        }

        Ok(())
    }

    #[allow(non_snake_case)]
    fn AcquireSRWLockShared(&mut self, lock_op: &OpTy<'tcx, Tag>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let id = srwlock_get_or_create_id(this, lock_op)?;
        let active_thread = this.get_active_thread();

        if this.rwlock_is_write_locked(id) {
            this.rwlock_enqueue_and_block_reader(id, active_thread);
        } else {
            this.rwlock_reader_lock(id, active_thread);
        }

        Ok(())
    }

    #[allow(non_snake_case)]
    fn TryAcquireSRWLockShared(&mut self, lock_op: &OpTy<'tcx, Tag>) -> InterpResult<'tcx, u8> {
        let this = self.eval_context_mut();
        let id = srwlock_get_or_create_id(this, lock_op)?;
        let active_thread = this.get_active_thread();

        if this.rwlock_is_write_locked(id) {
            Ok(0)
        } else {
            this.rwlock_reader_lock(id, active_thread);
            Ok(1)
        }
    }

    #[allow(non_snake_case)]
    fn ReleaseSRWLockShared(&mut self, lock_op: &OpTy<'tcx, Tag>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let id = srwlock_get_or_create_id(this, lock_op)?;
        let active_thread = this.get_active_thread();

        if !this.rwlock_reader_unlock(id, active_thread) {
            // The docs do not say anything about this case, but it seems better to not allow it.
            throw_ub_format!(
                "calling ReleaseSRWLockShared on an SRWLock that is not locked by the current thread"
            );
        }

        Ok(())
    }
}
