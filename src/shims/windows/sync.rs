use crate::*;

// Locks are pointer-sized pieces of data, initialized to 0.
// We use the first 4 bytes to store the RwLockId.

fn srwlock_get_or_create_id<'mir, 'tcx: 'mir>(
    ecx: &mut MiriInterpCx<'mir, 'tcx>,
    lock_op: &OpTy<'tcx, Provenance>,
) -> InterpResult<'tcx, RwLockId> {
    let value_place = ecx.deref_operand_and_offset(lock_op, 0, ecx.machine.layouts.u32)?;

    ecx.rwlock_get_or_create(|ecx, next_id| {
        let (old, success) = ecx
            .atomic_compare_exchange_scalar(
                &value_place,
                &ImmTy::from_uint(0u32, ecx.machine.layouts.u32),
                next_id.to_u32_scalar(),
                AtomicRwOrd::Relaxed,
                AtomicReadOrd::Relaxed,
                false,
            )?
            .to_scalar_pair();

        Ok(if success.to_bool().expect("compare_exchange's second return value is a bool") {
            // Caller of the closure needs to allocate next_id
            None
        } else {
            Some(RwLockId::from_u32(old.to_u32().expect("layout is u32")))
        })
    })
}

impl<'mir, 'tcx> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    #[allow(non_snake_case)]
    fn AcquireSRWLockExclusive(&mut self, lock_op: &OpTy<'tcx, Provenance>) -> InterpResult<'tcx> {
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
    fn TryAcquireSRWLockExclusive(
        &mut self,
        lock_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, u8> {
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
    fn ReleaseSRWLockExclusive(&mut self, lock_op: &OpTy<'tcx, Provenance>) -> InterpResult<'tcx> {
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
    fn AcquireSRWLockShared(&mut self, lock_op: &OpTy<'tcx, Provenance>) -> InterpResult<'tcx> {
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
    fn TryAcquireSRWLockShared(
        &mut self,
        lock_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, u8> {
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
    fn ReleaseSRWLockShared(&mut self, lock_op: &OpTy<'tcx, Provenance>) -> InterpResult<'tcx> {
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
