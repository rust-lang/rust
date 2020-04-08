use rustc_middle::ty::{TyKind, TypeAndMut};
use rustc_target::abi::{LayoutOf, Size};

use crate::stacked_borrows::Tag;
use crate::threads::{BlockSetId, ThreadId};
use crate::*;

fn assert_ptr_target_min_size<'mir, 'tcx: 'mir>(
    ecx: &MiriEvalContext<'mir, 'tcx>,
    operand: OpTy<'tcx, Tag>,
    min_size: u64,
) -> InterpResult<'tcx, ()> {
    let target_ty = match operand.layout.ty.kind {
        TyKind::RawPtr(TypeAndMut { ty, mutbl: _ }) => ty,
        _ => panic!("Argument to pthread function was not a raw pointer"),
    };
    let target_layout = ecx.layout_of(target_ty)?;
    assert!(target_layout.size.bytes() >= min_size);
    Ok(())
}

// pthread_mutexattr_t is either 4 or 8 bytes, depending on the platform.

// Our chosen memory layout for emulation (does not have to match the platform layout!):
// store an i32 in the first four bytes equal to the corresponding libc mutex kind constant
// (e.g. PTHREAD_MUTEX_NORMAL).

fn mutexattr_get_kind<'mir, 'tcx: 'mir>(
    ecx: &MiriEvalContext<'mir, 'tcx>,
    attr_op: OpTy<'tcx, Tag>,
) -> InterpResult<'tcx, ScalarMaybeUndef<Tag>> {
    // Ensure that the following read at an offset to the attr pointer is within bounds
    assert_ptr_target_min_size(ecx, attr_op, 4)?;
    let attr_place = ecx.deref_operand(attr_op)?;
    let kind_place =
        attr_place.offset(Size::ZERO, MemPlaceMeta::None, ecx.machine.layouts.i32, ecx)?;
    ecx.read_scalar(kind_place.into())
}

fn mutexattr_set_kind<'mir, 'tcx: 'mir>(
    ecx: &mut MiriEvalContext<'mir, 'tcx>,
    attr_op: OpTy<'tcx, Tag>,
    kind: impl Into<ScalarMaybeUndef<Tag>>,
) -> InterpResult<'tcx, ()> {
    // Ensure that the following write at an offset to the attr pointer is within bounds
    assert_ptr_target_min_size(ecx, attr_op, 4)?;
    let attr_place = ecx.deref_operand(attr_op)?;
    let kind_place =
        attr_place.offset(Size::ZERO, MemPlaceMeta::None, ecx.machine.layouts.i32, ecx)?;
    ecx.write_scalar(kind.into(), kind_place.into())
}

// pthread_mutex_t is between 24 and 48 bytes, depending on the platform.

// Our chosen memory layout for the emulated mutex (does not have to match the platform layout!):
// bytes 0-3: reserved for signature on macOS
// (need to avoid this because it is set by static initializer macros)
// bytes 4-7: count of how many times this mutex has been locked, as a u32
// bytes 8-11: when count > 0, id of the owner thread as a u32
// bytes 12-15 or 16-19 (depending on platform): mutex kind, as an i32
// (the kind has to be at its offset for compatibility with static initializer macros)
// bytes 20-23: when count > 0, id of the blockset in which the blocked threads are waiting.

fn mutex_get_locked_count<'mir, 'tcx: 'mir>(
    ecx: &MiriEvalContext<'mir, 'tcx>,
    mutex_op: OpTy<'tcx, Tag>,
) -> InterpResult<'tcx, ScalarMaybeUndef<Tag>> {
    // Ensure that the following read at an offset to the mutex pointer is within bounds
    assert_ptr_target_min_size(ecx, mutex_op, 24)?;
    let mutex_place = ecx.deref_operand(mutex_op)?;
    let locked_count_place = mutex_place.offset(
        Size::from_bytes(4),
        MemPlaceMeta::None,
        ecx.machine.layouts.u32,
        ecx,
    )?;
    ecx.read_scalar(locked_count_place.into())
}

fn mutex_set_locked_count<'mir, 'tcx: 'mir>(
    ecx: &mut MiriEvalContext<'mir, 'tcx>,
    mutex_op: OpTy<'tcx, Tag>,
    locked_count: impl Into<ScalarMaybeUndef<Tag>>,
) -> InterpResult<'tcx, ()> {
    // Ensure that the following write at an offset to the mutex pointer is within bounds
    assert_ptr_target_min_size(ecx, mutex_op, 24)?;
    let mutex_place = ecx.deref_operand(mutex_op)?;
    let locked_count_place = mutex_place.offset(
        Size::from_bytes(4),
        MemPlaceMeta::None,
        ecx.machine.layouts.u32,
        ecx,
    )?;
    ecx.write_scalar(locked_count.into(), locked_count_place.into())
}

fn mutex_get_owner<'mir, 'tcx: 'mir>(
    ecx: &MiriEvalContext<'mir, 'tcx>,
    mutex_op: OpTy<'tcx, Tag>,
) -> InterpResult<'tcx, ScalarMaybeUndef<Tag>> {
    // Ensure that the following read at an offset to the mutex pointer is within bounds
    assert_ptr_target_min_size(ecx, mutex_op, 24)?;
    let mutex_place = ecx.deref_operand(mutex_op)?;
    let mutex_id_place = mutex_place.offset(
        Size::from_bytes(8),
        MemPlaceMeta::None,
        ecx.machine.layouts.u32,
        ecx,
    )?;
    ecx.read_scalar(mutex_id_place.into())
}

fn mutex_set_owner<'mir, 'tcx: 'mir>(
    ecx: &mut MiriEvalContext<'mir, 'tcx>,
    mutex_op: OpTy<'tcx, Tag>,
    mutex_id: impl Into<ScalarMaybeUndef<Tag>>,
) -> InterpResult<'tcx, ()> {
    // Ensure that the following write at an offset to the mutex pointer is within bounds
    assert_ptr_target_min_size(ecx, mutex_op, 24)?;
    let mutex_place = ecx.deref_operand(mutex_op)?;
    let mutex_id_place = mutex_place.offset(
        Size::from_bytes(8),
        MemPlaceMeta::None,
        ecx.machine.layouts.u32,
        ecx,
    )?;
    ecx.write_scalar(mutex_id.into(), mutex_id_place.into())
}

fn mutex_get_kind<'mir, 'tcx: 'mir>(
    ecx: &mut MiriEvalContext<'mir, 'tcx>,
    mutex_op: OpTy<'tcx, Tag>,
) -> InterpResult<'tcx, ScalarMaybeUndef<Tag>> {
    // Ensure that the following read at an offset to the mutex pointer is within bounds
    assert_ptr_target_min_size(ecx, mutex_op, 24)?;
    let mutex_place = ecx.deref_operand(mutex_op)?;
    let kind_offset = if ecx.pointer_size().bytes() == 8 { 16 } else { 12 };
    let kind_place = mutex_place.offset(
        Size::from_bytes(kind_offset),
        MemPlaceMeta::None,
        ecx.machine.layouts.i32,
        ecx,
    )?;
    ecx.read_scalar(kind_place.into())
}

fn mutex_set_kind<'mir, 'tcx: 'mir>(
    ecx: &mut MiriEvalContext<'mir, 'tcx>,
    mutex_op: OpTy<'tcx, Tag>,
    kind: impl Into<ScalarMaybeUndef<Tag>>,
) -> InterpResult<'tcx, ()> {
    // Ensure that the following write at an offset to the mutex pointer is within bounds
    assert_ptr_target_min_size(ecx, mutex_op, 24)?;
    let mutex_place = ecx.deref_operand(mutex_op)?;
    let kind_offset = if ecx.pointer_size().bytes() == 8 { 16 } else { 12 };
    let kind_place = mutex_place.offset(
        Size::from_bytes(kind_offset),
        MemPlaceMeta::None,
        ecx.machine.layouts.i32,
        ecx,
    )?;
    ecx.write_scalar(kind.into(), kind_place.into())
}

fn mutex_get_blockset<'mir, 'tcx: 'mir>(
    ecx: &MiriEvalContext<'mir, 'tcx>,
    mutex_op: OpTy<'tcx, Tag>,
) -> InterpResult<'tcx, ScalarMaybeUndef<Tag>> {
    // Ensure that the following read at an offset to the mutex pointer is within bounds
    assert_ptr_target_min_size(ecx, mutex_op, 24)?;
    let mutex_place = ecx.deref_operand(mutex_op)?;
    let mutex_id_place = mutex_place.offset(
        Size::from_bytes(20),
        MemPlaceMeta::None,
        ecx.machine.layouts.u32,
        ecx,
    )?;
    ecx.read_scalar(mutex_id_place.into())
}

fn mutex_set_blockset<'mir, 'tcx: 'mir>(
    ecx: &mut MiriEvalContext<'mir, 'tcx>,
    mutex_op: OpTy<'tcx, Tag>,
    mutex_id: impl Into<ScalarMaybeUndef<Tag>>,
) -> InterpResult<'tcx, ()> {
    // Ensure that the following write at an offset to the mutex pointer is within bounds
    assert_ptr_target_min_size(ecx, mutex_op, 24)?;
    let mutex_place = ecx.deref_operand(mutex_op)?;
    let mutex_id_place = mutex_place.offset(
        Size::from_bytes(20),
        MemPlaceMeta::None,
        ecx.machine.layouts.u32,
        ecx,
    )?;
    ecx.write_scalar(mutex_id.into(), mutex_id_place.into())
}

fn mutex_get_or_create_blockset<'mir, 'tcx: 'mir>(
    ecx: &mut MiriEvalContext<'mir, 'tcx>,
    mutex_op: OpTy<'tcx, Tag>,
) -> InterpResult<'tcx, BlockSetId> {
    let blockset = mutex_get_blockset(ecx, mutex_op)?.to_u32()?;
    if blockset == 0 {
        // 0 is a default value and also not a valid blockset id. Need to
        // allocate a new blockset.
        let blockset = ecx.create_blockset()?;
        mutex_set_blockset(ecx, mutex_op, blockset.to_u32_scalar())?;
        Ok(blockset)
    } else {
        Ok(blockset.into())
    }
}

// pthread_rwlock_t is between 32 and 56 bytes, depending on the platform.

// Our chosen memory layout for the emulated rwlock (does not have to match the platform layout!):
// bytes 0-3: reserved for signature on macOS
// (need to avoid this because it is set by static initializer macros)
// bytes 4-7: reader count, as a u32
// bytes 8-11: writer count, as a u32
// bytes 12-15: when writer or reader count > 0, id of the blockset in which the
// blocked writers are waiting.
// bytes 16-20: when writer count > 0, id of the blockset in which the blocked
// readers are waiting.

fn rwlock_get_readers<'mir, 'tcx: 'mir>(
    ecx: &MiriEvalContext<'mir, 'tcx>,
    rwlock_op: OpTy<'tcx, Tag>,
) -> InterpResult<'tcx, ScalarMaybeUndef<Tag>> {
    // Ensure that the following read at an offset to the rwlock pointer is within bounds
    assert_ptr_target_min_size(ecx, rwlock_op, 20)?;
    let rwlock_place = ecx.deref_operand(rwlock_op)?;
    let readers_place = rwlock_place.offset(
        Size::from_bytes(4),
        MemPlaceMeta::None,
        ecx.machine.layouts.u32,
        ecx,
    )?;
    ecx.read_scalar(readers_place.into())
}

fn rwlock_set_readers<'mir, 'tcx: 'mir>(
    ecx: &mut MiriEvalContext<'mir, 'tcx>,
    rwlock_op: OpTy<'tcx, Tag>,
    readers: impl Into<ScalarMaybeUndef<Tag>>,
) -> InterpResult<'tcx, ()> {
    // Ensure that the following write at an offset to the rwlock pointer is within bounds
    assert_ptr_target_min_size(ecx, rwlock_op, 20)?;
    let rwlock_place = ecx.deref_operand(rwlock_op)?;
    let readers_place = rwlock_place.offset(
        Size::from_bytes(4),
        MemPlaceMeta::None,
        ecx.machine.layouts.u32,
        ecx,
    )?;
    ecx.write_scalar(readers.into(), readers_place.into())
}

fn rwlock_get_writers<'mir, 'tcx: 'mir>(
    ecx: &MiriEvalContext<'mir, 'tcx>,
    rwlock_op: OpTy<'tcx, Tag>,
) -> InterpResult<'tcx, ScalarMaybeUndef<Tag>> {
    // Ensure that the following read at an offset to the rwlock pointer is within bounds
    assert_ptr_target_min_size(ecx, rwlock_op, 20)?;
    let rwlock_place = ecx.deref_operand(rwlock_op)?;
    let writers_place = rwlock_place.offset(
        Size::from_bytes(8),
        MemPlaceMeta::None,
        ecx.machine.layouts.u32,
        ecx,
    )?;
    ecx.read_scalar(writers_place.into())
}

fn rwlock_set_writers<'mir, 'tcx: 'mir>(
    ecx: &mut MiriEvalContext<'mir, 'tcx>,
    rwlock_op: OpTy<'tcx, Tag>,
    writers: impl Into<ScalarMaybeUndef<Tag>>,
) -> InterpResult<'tcx, ()> {
    // Ensure that the following write at an offset to the rwlock pointer is within bounds
    assert_ptr_target_min_size(ecx, rwlock_op, 20)?;
    let rwlock_place = ecx.deref_operand(rwlock_op)?;
    let writers_place = rwlock_place.offset(
        Size::from_bytes(8),
        MemPlaceMeta::None,
        ecx.machine.layouts.u32,
        ecx,
    )?;
    ecx.write_scalar(writers.into(), writers_place.into())
}

fn rwlock_get_writer_blockset<'mir, 'tcx: 'mir>(
    ecx: &MiriEvalContext<'mir, 'tcx>,
    rwlock_op: OpTy<'tcx, Tag>,
) -> InterpResult<'tcx, ScalarMaybeUndef<Tag>> {
    // Ensure that the following read at an offset to the rwlock pointer is within bounds
    assert_ptr_target_min_size(ecx, rwlock_op, 20)?;
    let rwlock_place = ecx.deref_operand(rwlock_op)?;
    let blockset_place = rwlock_place.offset(
        Size::from_bytes(12),
        MemPlaceMeta::None,
        ecx.machine.layouts.u32,
        ecx,
    )?;
    ecx.read_scalar(blockset_place.into())
}

fn rwlock_set_writer_blockset<'mir, 'tcx: 'mir>(
    ecx: &mut MiriEvalContext<'mir, 'tcx>,
    rwlock_op: OpTy<'tcx, Tag>,
    blockset: impl Into<ScalarMaybeUndef<Tag>>,
) -> InterpResult<'tcx, ()> {
    // Ensure that the following write at an offset to the rwlock pointer is within bounds
    assert_ptr_target_min_size(ecx, rwlock_op, 20)?;
    let rwlock_place = ecx.deref_operand(rwlock_op)?;
    let blockset_place = rwlock_place.offset(
        Size::from_bytes(12),
        MemPlaceMeta::None,
        ecx.machine.layouts.u32,
        ecx,
    )?;
    ecx.write_scalar(blockset.into(), blockset_place.into())
}

fn rwlock_get_or_create_writer_blockset<'mir, 'tcx: 'mir>(
    ecx: &mut MiriEvalContext<'mir, 'tcx>,
    rwlock_op: OpTy<'tcx, Tag>,
) -> InterpResult<'tcx, BlockSetId> {
    let blockset = rwlock_get_writer_blockset(ecx, rwlock_op)?.to_u32()?;
    if blockset == 0 {
        // 0 is a default value and also not a valid blockset id. Need to
        // allocate a new blockset.
        let blockset = ecx.create_blockset()?;
        rwlock_set_writer_blockset(ecx, rwlock_op, blockset.to_u32_scalar())?;
        Ok(blockset)
    } else {
        Ok(blockset.into())
    }
}

fn rwlock_get_reader_blockset<'mir, 'tcx: 'mir>(
    ecx: &MiriEvalContext<'mir, 'tcx>,
    rwlock_op: OpTy<'tcx, Tag>,
) -> InterpResult<'tcx, ScalarMaybeUndef<Tag>> {
    // Ensure that the following read at an offset to the rwlock pointer is within bounds
    assert_ptr_target_min_size(ecx, rwlock_op, 20)?;
    let rwlock_place = ecx.deref_operand(rwlock_op)?;
    let blockset_place = rwlock_place.offset(
        Size::from_bytes(16),
        MemPlaceMeta::None,
        ecx.machine.layouts.u32,
        ecx,
    )?;
    ecx.read_scalar(blockset_place.into())
}

fn rwlock_set_reader_blockset<'mir, 'tcx: 'mir>(
    ecx: &mut MiriEvalContext<'mir, 'tcx>,
    rwlock_op: OpTy<'tcx, Tag>,
    blockset: impl Into<ScalarMaybeUndef<Tag>>,
) -> InterpResult<'tcx, ()> {
    // Ensure that the following write at an offset to the rwlock pointer is within bounds
    assert_ptr_target_min_size(ecx, rwlock_op, 20)?;
    let rwlock_place = ecx.deref_operand(rwlock_op)?;
    let blockset_place = rwlock_place.offset(
        Size::from_bytes(16),
        MemPlaceMeta::None,
        ecx.machine.layouts.u32,
        ecx,
    )?;
    ecx.write_scalar(blockset.into(), blockset_place.into())
}

fn rwlock_get_or_create_reader_blockset<'mir, 'tcx: 'mir>(
    ecx: &mut MiriEvalContext<'mir, 'tcx>,
    rwlock_op: OpTy<'tcx, Tag>,
) -> InterpResult<'tcx, BlockSetId> {
    let blockset = rwlock_get_reader_blockset(ecx, rwlock_op)?.to_u32()?;
    if blockset == 0 {
        // 0 is a default value and also not a valid blockset id. Need to
        // allocate a new blockset.
        let blockset = ecx.create_blockset()?;
        rwlock_set_reader_blockset(ecx, rwlock_op, blockset.to_u32_scalar())?;
        Ok(blockset)
    } else {
        Ok(blockset.into())
    }
}

impl<'mir, 'tcx> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn pthread_mutexattr_init(&mut self, attr_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let default_kind = this.eval_libc("PTHREAD_MUTEX_DEFAULT")?;
        mutexattr_set_kind(this, attr_op, default_kind)?;

        Ok(0)
    }

    fn pthread_mutexattr_settype(
        &mut self,
        attr_op: OpTy<'tcx, Tag>,
        kind_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let kind = this.read_scalar(kind_op)?.not_undef()?;
        if kind == this.eval_libc("PTHREAD_MUTEX_NORMAL")?
            || kind == this.eval_libc("PTHREAD_MUTEX_ERRORCHECK")?
            || kind == this.eval_libc("PTHREAD_MUTEX_RECURSIVE")?
        {
            mutexattr_set_kind(this, attr_op, kind)?;
        } else {
            let einval = this.eval_libc_i32("EINVAL")?;
            return Ok(einval);
        }

        Ok(0)
    }

    fn pthread_mutexattr_destroy(&mut self, attr_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        mutexattr_set_kind(this, attr_op, ScalarMaybeUndef::Undef)?;

        Ok(0)
    }

    fn pthread_mutex_init(
        &mut self,
        mutex_op: OpTy<'tcx, Tag>,
        attr_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let attr = this.read_scalar(attr_op)?.not_undef()?;
        let kind = if this.is_null(attr)? {
            this.eval_libc("PTHREAD_MUTEX_DEFAULT")?
        } else {
            mutexattr_get_kind(this, attr_op)?.not_undef()?
        };

        mutex_set_locked_count(this, mutex_op, Scalar::from_u32(0))?;
        mutex_set_kind(this, mutex_op, kind)?;

        Ok(0)
    }

    fn pthread_mutex_lock(&mut self, mutex_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let kind = mutex_get_kind(this, mutex_op)?.not_undef()?;
        let locked_count = mutex_get_locked_count(this, mutex_op)?.to_u32()?;
        let active_thread = this.get_active_thread()?;

        if locked_count == 0 {
            // The mutex is unlocked. Let's lock it.
            mutex_set_locked_count(this, mutex_op, Scalar::from_u32(1))?;
            mutex_set_owner(this, mutex_op, active_thread.to_u32_scalar())?;
            Ok(0)
        } else {
            // The mutex is locked. Let's check by whom.
            let owner_thread: ThreadId =
                mutex_get_owner(this, mutex_op)?.not_undef()?.to_u32()?.into();
            if owner_thread != active_thread {
                // Block the active thread.
                let blockset = mutex_get_or_create_blockset(this, mutex_op)?;
                this.block_active_thread(blockset)?;
                Ok(0)
            } else {
                // Trying to acquire the same mutex again.
                if kind == this.eval_libc("PTHREAD_MUTEX_NORMAL")? {
                    throw_machine_stop!(TerminationInfo::Deadlock);
                } else if kind == this.eval_libc("PTHREAD_MUTEX_ERRORCHECK")? {
                    this.eval_libc_i32("EDEADLK")
                } else if kind == this.eval_libc("PTHREAD_MUTEX_RECURSIVE")? {
                    match locked_count.checked_add(1) {
                        Some(new_count) => {
                            mutex_set_locked_count(this, mutex_op, Scalar::from_u32(new_count))?;
                            Ok(0)
                        }
                        None => this.eval_libc_i32("EAGAIN"),
                    }
                } else {
                    throw_ub_format!("called pthread_mutex_lock on an unsupported type of mutex");
                }
            }
        }
    }

    fn pthread_mutex_trylock(&mut self, mutex_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let kind = mutex_get_kind(this, mutex_op)?.not_undef()?;
        let locked_count = mutex_get_locked_count(this, mutex_op)?.to_u32()?;
        let active_thread = this.get_active_thread()?;

        if locked_count == 0 {
            // The mutex is unlocked. Let's lock it.
            mutex_set_locked_count(this, mutex_op, Scalar::from_u32(1))?;
            mutex_set_owner(this, mutex_op, active_thread.to_u32_scalar())?;
            Ok(0)
        } else {
            let owner_thread: ThreadId = mutex_get_owner(this, mutex_op)?.to_u32()?.into();
            if owner_thread != active_thread {
                this.eval_libc_i32("EBUSY")
            } else {
                if kind == this.eval_libc("PTHREAD_MUTEX_NORMAL")?
                    || kind == this.eval_libc("PTHREAD_MUTEX_ERRORCHECK")?
                {
                    this.eval_libc_i32("EBUSY")
                } else if kind == this.eval_libc("PTHREAD_MUTEX_RECURSIVE")? {
                    match locked_count.checked_add(1) {
                        Some(new_count) => {
                            mutex_set_locked_count(this, mutex_op, Scalar::from_u32(new_count))?;
                            Ok(0)
                        }
                        None => this.eval_libc_i32("EAGAIN"),
                    }
                } else {
                    throw_ub_format!(
                        "called pthread_mutex_trylock on an unsupported type of mutex"
                    );
                }
            }
        }
    }

    fn pthread_mutex_unlock(&mut self, mutex_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let kind = mutex_get_kind(this, mutex_op)?.not_undef()?;
        let locked_count = mutex_get_locked_count(this, mutex_op)?.to_u32()?;
        let owner_thread: ThreadId = mutex_get_owner(this, mutex_op)?.to_u32()?.into();

        if owner_thread != this.get_active_thread()? {
            throw_ub_format!("called pthread_mutex_unlock on a mutex owned by another thread");
        } else if locked_count == 1 {
            let blockset = mutex_get_or_create_blockset(this, mutex_op)?;
            if let Some(new_owner) = this.unblock_random_thread(blockset)? {
                // We have at least one thread waiting on this mutex. Transfer
                // ownership to it.
                mutex_set_owner(this, mutex_op, new_owner.to_u32_scalar())?;
            } else {
                // No thread is waiting on this mutex.
                mutex_set_locked_count(this, mutex_op, Scalar::from_u32(0))?;
            }
            Ok(0)
        } else {
            if kind == this.eval_libc("PTHREAD_MUTEX_NORMAL")? {
                throw_ub_format!("unlocked a PTHREAD_MUTEX_NORMAL mutex that was not locked");
            } else if kind == this.eval_libc("PTHREAD_MUTEX_ERRORCHECK")? {
                this.eval_libc_i32("EPERM")
            } else if kind == this.eval_libc("PTHREAD_MUTEX_RECURSIVE")? {
                match locked_count.checked_sub(1) {
                    Some(new_count) => {
                        mutex_set_locked_count(this, mutex_op, Scalar::from_u32(new_count))?;
                        Ok(0)
                    }
                    None => {
                        // locked_count was already zero
                        this.eval_libc_i32("EPERM")
                    }
                }
            } else {
                throw_ub_format!("called pthread_mutex_unlock on an unsupported type of mutex");
            }
        }
    }

    fn pthread_mutex_destroy(&mut self, mutex_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        if mutex_get_locked_count(this, mutex_op)?.to_u32()? != 0 {
            throw_ub_format!("destroyed a locked mutex");
        }

        mutex_set_kind(this, mutex_op, ScalarMaybeUndef::Undef)?;
        mutex_set_locked_count(this, mutex_op, ScalarMaybeUndef::Undef)?;
        mutex_set_blockset(this, mutex_op, ScalarMaybeUndef::Undef)?;

        Ok(0)
    }

    fn pthread_rwlock_rdlock(&mut self, rwlock_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let readers = rwlock_get_readers(this, rwlock_op)?.to_u32()?;
        let writers = rwlock_get_writers(this, rwlock_op)?.to_u32()?;

        if writers != 0 {
            // The lock is locked by a writer.
            assert_eq!(writers, 1);
            let reader_blockset = rwlock_get_or_create_reader_blockset(this, rwlock_op)?;
            this.block_active_thread(reader_blockset)?;
            Ok(0)
        } else {
            match readers.checked_add(1) {
                Some(new_readers) => {
                    rwlock_set_readers(this, rwlock_op, Scalar::from_u32(new_readers))?;
                    Ok(0)
                }
                None => this.eval_libc_i32("EAGAIN"),
            }
        }
    }

    fn pthread_rwlock_tryrdlock(&mut self, rwlock_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let readers = rwlock_get_readers(this, rwlock_op)?.to_u32()?;
        let writers = rwlock_get_writers(this, rwlock_op)?.to_u32()?;
        if writers != 0 {
            this.eval_libc_i32("EBUSY")
        } else {
            match readers.checked_add(1) {
                Some(new_readers) => {
                    rwlock_set_readers(this, rwlock_op, Scalar::from_u32(new_readers))?;
                    Ok(0)
                }
                None => this.eval_libc_i32("EAGAIN"),
            }
        }
    }

    fn pthread_rwlock_wrlock(&mut self, rwlock_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let readers = rwlock_get_readers(this, rwlock_op)?.to_u32()?;
        let writers = rwlock_get_writers(this, rwlock_op)?.to_u32()?;
        let writer_blockset = rwlock_get_or_create_writer_blockset(this, rwlock_op)?;
        if readers != 0 || writers != 0 {
            this.block_active_thread(writer_blockset)?;
        } else {
            rwlock_set_writers(this, rwlock_op, Scalar::from_u32(1))?;
        }
        Ok(0)
    }

    fn pthread_rwlock_trywrlock(&mut self, rwlock_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let readers = rwlock_get_readers(this, rwlock_op)?.to_u32()?;
        let writers = rwlock_get_writers(this, rwlock_op)?.to_u32()?;
        if readers != 0 || writers != 0 {
            this.eval_libc_i32("EBUSY")
        } else {
            rwlock_set_writers(this, rwlock_op, Scalar::from_u32(1))?;
            Ok(0)
        }
    }

    fn pthread_rwlock_unlock(&mut self, rwlock_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let readers = rwlock_get_readers(this, rwlock_op)?.to_u32()?;
        let writers = rwlock_get_writers(this, rwlock_op)?.to_u32()?;
        let writer_blockset = rwlock_get_or_create_writer_blockset(this, rwlock_op)?;
        if let Some(new_readers) = readers.checked_sub(1) {
            assert_eq!(writers, 0);
            rwlock_set_readers(this, rwlock_op, Scalar::from_u32(new_readers))?;
            if new_readers == 0 {
                if let Some(_writer) = this.unblock_random_thread(writer_blockset)? {
                    rwlock_set_writers(this, rwlock_op, Scalar::from_u32(1))?;
                }
            }
            Ok(0)
        } else if writers != 0 {
            let reader_blockset = rwlock_get_or_create_reader_blockset(this, rwlock_op)?;
            rwlock_set_writers(this, rwlock_op, Scalar::from_u32(0))?;
            if let Some(_writer) = this.unblock_random_thread(writer_blockset)? {
                rwlock_set_writers(this, rwlock_op, Scalar::from_u32(1))?;
            } else {
                let mut readers = 0;
                while let Some(_reader) = this.unblock_random_thread(reader_blockset)? {
                    readers += 1;
                }
                rwlock_set_readers(this, rwlock_op, Scalar::from_u32(readers))?
            }
            Ok(0)
        } else {
            throw_ub_format!("unlocked an rwlock that was not locked");
        }
    }

    fn pthread_rwlock_destroy(&mut self, rwlock_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        if rwlock_get_readers(this, rwlock_op)?.to_u32()? != 0
            || rwlock_get_writers(this, rwlock_op)?.to_u32()? != 0
        {
            throw_ub_format!("destroyed a locked rwlock");
        }

        rwlock_set_readers(this, rwlock_op, ScalarMaybeUndef::Undef)?;
        rwlock_set_writers(this, rwlock_op, ScalarMaybeUndef::Undef)?;
        rwlock_set_reader_blockset(this, rwlock_op, ScalarMaybeUndef::Undef)?;
        rwlock_set_writer_blockset(this, rwlock_op, ScalarMaybeUndef::Undef)?;

        Ok(0)
    }
}
