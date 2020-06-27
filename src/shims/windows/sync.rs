use rustc_target::abi::Size;

use crate::*;

// Locks are pointer-sized pieces of data, initialized to 0.
// We use them to count readers, with usize::MAX representing the write-locked state.

fn deref_lock<'mir, 'tcx: 'mir>(
    ecx: &mut MiriEvalContext<'mir, 'tcx>,
    lock_op: OpTy<'tcx, Tag>,
) -> InterpResult<'tcx, MPlaceTy<'tcx, Tag>> {
    // `lock` is a pointer to `void*`; cast it to a pointer to `usize`.
    let lock = ecx.deref_operand(lock_op)?;
    let usize = ecx.machine.layouts.usize;
    assert_eq!(lock.layout.size, usize.size);
    Ok(lock.offset(Size::ZERO, MemPlaceMeta::None, usize, ecx)?)
}

impl<'mir, 'tcx> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    #[allow(non_snake_case)]
    fn AcquireSRWLockExclusive(
        &mut self,
        lock_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        assert_eq!(this.get_total_thread_count(), 1, "concurrency on Windows is not supported");

        let lock = deref_lock(this, lock_op)?;
        let lock_val = this.read_scalar(lock.into())?.to_machine_usize(this)?;
        if lock_val == 0 {
            // Currently not locked. Lock it.
            let new_val = Scalar::from_machine_usize(this.machine_usize_max(), this);
            this.write_scalar(new_val, lock.into())?;
        } else {
            // Lock is already held. This is a deadlock.
            throw_machine_stop!(TerminationInfo::Deadlock);
        }

        Ok(())
    }

    #[allow(non_snake_case)]
    fn TryAcquireSRWLockExclusive(
        &mut self,
        lock_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, u8> {
        let this = self.eval_context_mut();
        assert_eq!(this.get_total_thread_count(), 1, "concurrency on Windows is not supported");

        let lock = deref_lock(this, lock_op)?;
        let lock_val = this.read_scalar(lock.into())?.to_machine_usize(this)?;
        if lock_val == 0 {
            // Currently not locked. Lock it.
            let new_val = this.machine_usize_max();
            this.write_scalar(Scalar::from_machine_usize(new_val, this), lock.into())?;
            Ok(1)
        } else {
            // Lock is already held.
            Ok(0)
        }
    }

    #[allow(non_snake_case)]
    fn ReleaseSRWLockExclusive(
        &mut self,
        lock_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        assert_eq!(this.get_total_thread_count(), 1, "concurrency on Windows is not supported");

        let lock = deref_lock(this, lock_op)?;
        let lock_val = this.read_scalar(lock.into())?.to_machine_usize(this)?;
        if lock_val == this.machine_usize_max() {
            // Currently locked. Unlock it.
            let new_val = 0;
            this.write_scalar(Scalar::from_machine_usize(new_val, this), lock.into())?;
        } else {
            // Lock is not locked.
            throw_ub_format!("calling ReleaseSRWLockExclusive on an SRWLock that is not exclusively locked");
        }

        Ok(())
    }

    #[allow(non_snake_case)]
    fn AcquireSRWLockShared(
        &mut self,
        lock_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        assert_eq!(this.get_total_thread_count(), 1, "concurrency on Windows is not supported");

        let lock = deref_lock(this, lock_op)?;
        let lock_val = this.read_scalar(lock.into())?.to_machine_usize(this)?;
        if lock_val == this.machine_usize_max() {
            // Currently write locked. This is a deadlock.
            throw_machine_stop!(TerminationInfo::Deadlock);
        } else {
            // Bump up read counter (cannot overflow as we just checkd against usize::MAX);
            let new_val = lock_val+1;
            // Make sure this does not reach the "write locked" flag.
            if new_val == this.machine_usize_max() {
                throw_unsup_format!("SRWLock read-acquired too many times");
            }
            this.write_scalar(Scalar::from_machine_usize(new_val, this), lock.into())?;
        }

        Ok(())
    }

    #[allow(non_snake_case)]
    fn TryAcquireSRWLockShared(
        &mut self,
        lock_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, u8> {
        let this = self.eval_context_mut();
        assert_eq!(this.get_total_thread_count(), 1, "concurrency on Windows is not supported");

        let lock = deref_lock(this, lock_op)?;
        let lock_val = this.read_scalar(lock.into())?.to_machine_usize(this)?;
        if lock_val == this.machine_usize_max() {
            // Currently write locked.
            Ok(0)
        } else {
            // Bump up read counter (cannot overflow as we just checkd against usize::MAX);
            let new_val = lock_val+1;
            // Make sure this does not reach the "write locked" flag.
            if new_val == this.machine_usize_max() {
                throw_unsup_format!("SRWLock read-acquired too many times");
            }
            this.write_scalar(Scalar::from_machine_usize(new_val, this), lock.into())?;
            Ok(1)
        }
    }

    #[allow(non_snake_case)]
    fn ReleaseSRWLockShared(
        &mut self,
        lock_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        assert_eq!(this.get_total_thread_count(), 1, "concurrency on Windows is not supported");

        let lock = deref_lock(this, lock_op)?;
        let lock_val = this.read_scalar(lock.into())?.to_machine_usize(this)?;
        if lock_val == this.machine_usize_max() {
            // Currently write locked. This is a UB.
            throw_ub_format!("calling ReleaseSRWLockShared on write-locked SRWLock");
        } else if lock_val == 0 {
            // Currently not locked at all.
            throw_ub_format!("calling ReleaseSRWLockShared on unlocked SRWLock");
        } else {
            // Decrement read counter (cannot overflow as we just checkd against 0);
            let new_val = lock_val-1;
            this.write_scalar(Scalar::from_machine_usize(new_val, this), lock.into())?;
        }

        Ok(())
    }
}
