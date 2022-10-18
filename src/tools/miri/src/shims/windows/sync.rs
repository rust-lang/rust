use crate::concurrency::sync::InitOnceStatus;
use crate::concurrency::thread::MachineCallback;
use crate::*;

impl<'mir, 'tcx> EvalContextExtPriv<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
trait EvalContextExtPriv<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    // These synchronization structures are pointer-sized pieces of data, initialized to 0.
    // We use the first 4 bytes to store the id.
    fn get_or_create_id(
        &mut self,
        next_id: Scalar<Provenance>,
        lock_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, Option<u32>> {
        let this = self.eval_context_mut();
        let value_place = this.deref_operand_and_offset(lock_op, 0, this.machine.layouts.u32)?;

        let (old, success) = this
            .atomic_compare_exchange_scalar(
                &value_place,
                &ImmTy::from_uint(0u32, this.machine.layouts.u32),
                next_id,
                AtomicRwOrd::Relaxed,
                AtomicReadOrd::Relaxed,
                false,
            )?
            .to_scalar_pair();

        Ok(if success.to_bool().expect("compare_exchange's second return value is a bool") {
            // Caller of the closure needs to allocate next_id
            None
        } else {
            Some(old.to_u32().expect("layout is u32"))
        })
    }

    fn srwlock_get_or_create_id(
        &mut self,
        lock_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, RwLockId> {
        let this = self.eval_context_mut();
        this.rwlock_get_or_create(|ecx, next_id| {
            Ok(ecx.get_or_create_id(next_id.to_u32_scalar(), lock_op)?.map(RwLockId::from_u32))
        })
    }

    fn init_once_get_or_create_id(
        &mut self,
        lock_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, InitOnceId> {
        let this = self.eval_context_mut();
        this.init_once_get_or_create(|ecx, next_id| {
            Ok(ecx.get_or_create_id(next_id.to_u32_scalar(), lock_op)?.map(InitOnceId::from_u32))
        })
    }
}

impl<'mir, 'tcx> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}

#[allow(non_snake_case)]
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    fn AcquireSRWLockExclusive(&mut self, lock_op: &OpTy<'tcx, Provenance>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let id = this.srwlock_get_or_create_id(lock_op)?;
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
        let id = this.srwlock_get_or_create_id(lock_op)?;
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
        let id = this.srwlock_get_or_create_id(lock_op)?;
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
        let id = this.srwlock_get_or_create_id(lock_op)?;
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
        let id = this.srwlock_get_or_create_id(lock_op)?;
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
        let id = this.srwlock_get_or_create_id(lock_op)?;
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

        let id = this.init_once_get_or_create_id(init_once_op)?;
        let flags = this.read_scalar(flags_op)?.to_u32()?;
        let pending_place = this.deref_operand(pending_op)?.into();
        let context = this.read_pointer(context_op)?;

        if flags != 0 {
            throw_unsup_format!("unsupported `dwFlags` {flags} in `InitOnceBeginInitialize`");
        }

        if !this.ptr_is_null(context)? {
            throw_unsup_format!("non-null `lpContext` in `InitOnceBeginInitialize`");
        }

        struct Callback<'tcx> {
            init_once_id: InitOnceId,
            pending_place: PlaceTy<'tcx, Provenance>,
        }

        impl<'tcx> VisitTags for Callback<'tcx> {
            fn visit_tags(&self, visit: &mut dyn FnMut(SbTag)) {
                let Callback { init_once_id: _, pending_place } = self;
                pending_place.visit_tags(visit);
            }
        }

        impl<'mir, 'tcx> MachineCallback<'mir, 'tcx> for Callback<'tcx> {
            fn call(&self, this: &mut MiriInterpCx<'mir, 'tcx>) -> InterpResult<'tcx> {
                let pending = match this.init_once_status(self.init_once_id) {
                    InitOnceStatus::Uninitialized =>
                        unreachable!("status should have either been set to begun or complete"),
                    InitOnceStatus::Begun => this.eval_windows("c", "TRUE")?,
                    InitOnceStatus::Complete => this.eval_windows("c", "FALSE")?,
                };

                this.write_scalar(pending, &self.pending_place)?;

                Ok(())
            }
        }

        match this.init_once_status(id) {
            InitOnceStatus::Uninitialized => {
                this.init_once_begin(id);
                this.write_scalar(this.eval_windows("c", "TRUE")?, &pending_place)?;
            }
            InitOnceStatus::Begun =>
                this.init_once_enqueue_and_block(
                    id,
                    active_thread,
                    Box::new(Callback { init_once_id: id, pending_place }),
                ),
            InitOnceStatus::Complete =>
                this.write_scalar(this.eval_windows("c", "FALSE")?, &pending_place)?,
        }

        Ok(Scalar::from_i32(1))
    }

    fn InitOnceComplete(
        &mut self,
        init_once_op: &OpTy<'tcx, Provenance>,
        flags_op: &OpTy<'tcx, Provenance>,
        context_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, Scalar<Provenance>> {
        let this = self.eval_context_mut();

        let id = this.init_once_get_or_create_id(init_once_op)?;
        let flags = this.read_scalar(flags_op)?.to_u32()?;
        let context = this.read_pointer(context_op)?;

        let success = if flags == 0 {
            true
        } else if flags == this.eval_windows("c", "INIT_ONCE_INIT_FAILED")?.to_u32()? {
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

        Ok(Scalar::from_i32(1))
    }
}
