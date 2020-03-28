use std::sync::atomic::{AtomicU64, Ordering};

use rustc_middle::ty::{TyKind, TypeAndMut};
use rustc_target::abi::{FieldsShape, LayoutOf, Size};

use crate::stacked_borrows::Tag;
use crate::*;

impl<'mir, 'tcx> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn pthread_mutexattr_init(&mut self, attr_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let attr = this.read_scalar(attr_op)?.not_undef()?;
        if this.is_null(attr)? {
            return this.eval_libc_i32("EINVAL");
        }

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

        let attr = this.read_scalar(attr_op)?.not_undef()?;
        if this.is_null(attr)? {
            return this.eval_libc_i32("EINVAL");
        }

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

        let attr = this.read_scalar(attr_op)?.not_undef()?;
        if this.is_null(attr)? {
            return this.eval_libc_i32("EINVAL");
        }

        mutexattr_set_kind(this, attr_op, ScalarMaybeUndef::Undef)?;

        Ok(0)
    }

    fn pthread_mutex_init(
        &mut self,
        mutex_op: OpTy<'tcx, Tag>,
        attr_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let mutex = this.read_scalar(mutex_op)?.not_undef()?;
        if this.is_null(mutex)? {
            return this.eval_libc_i32("EINVAL");
        }

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

        let mutex = this.read_scalar(mutex_op)?.not_undef()?;
        if this.is_null(mutex)? {
            return this.eval_libc_i32("EINVAL");
        }

        let kind = mutex_get_kind(this, mutex_op)?.not_undef()?;
        let locked_count = mutex_get_locked_count(this, mutex_op)?.to_u32()?;

        if kind == this.eval_libc("PTHREAD_MUTEX_NORMAL")? {
            if locked_count == 0 {
                mutex_set_locked_count(this, mutex_op, Scalar::from_u32(1))?;
                Ok(0)
            } else {
                throw_unsup_format!("Deadlock due to locking a PTHREAD_MUTEX_NORMAL mutex twice");
            }
        } else if kind == this.eval_libc("PTHREAD_MUTEX_ERRORCHECK")? {
            if locked_count == 0 {
                mutex_set_locked_count(this, mutex_op, Scalar::from_u32(1))?;
                Ok(0)
            } else {
                this.eval_libc_i32("EDEADLK")
            }
        } else if kind == this.eval_libc("PTHREAD_MUTEX_RECURSIVE")? {
            match locked_count.checked_add(1) {
                Some(new_count) => {
                    mutex_set_locked_count(this, mutex_op, Scalar::from_u32(new_count))?;
                    Ok(0)
                }
                None => this.eval_libc_i32("EAGAIN"),
            }
        } else {
            this.eval_libc_i32("EINVAL")
        }
    }

    fn pthread_mutex_trylock(&mut self, mutex_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let mutex = this.read_scalar(mutex_op)?.not_undef()?;
        if this.is_null(mutex)? {
            return this.eval_libc_i32("EINVAL");
        }

        let kind = mutex_get_kind(this, mutex_op)?.not_undef()?;
        let locked_count = mutex_get_locked_count(this, mutex_op)?.to_u32()?;

        if kind == this.eval_libc("PTHREAD_MUTEX_NORMAL")?
            || kind == this.eval_libc("PTHREAD_MUTEX_ERRORCHECK")?
        {
            if locked_count == 0 {
                mutex_set_locked_count(this, mutex_op, Scalar::from_u32(1))?;
                Ok(0)
            } else {
                this.eval_libc_i32("EBUSY")
            }
        } else if kind == this.eval_libc("PTHREAD_MUTEX_RECURSIVE")? {
            match locked_count.checked_add(1) {
                Some(new_count) => {
                    mutex_set_locked_count(this, mutex_op, Scalar::from_u32(new_count))?;
                    Ok(0)
                }
                None => this.eval_libc_i32("EAGAIN"),
            }
        } else {
            this.eval_libc_i32("EINVAL")
        }
    }

    fn pthread_mutex_unlock(&mut self, mutex_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let mutex = this.read_scalar(mutex_op)?.not_undef()?;
        if this.is_null(mutex)? {
            return this.eval_libc_i32("EINVAL");
        }

        let kind = mutex_get_kind(this, mutex_op)?.not_undef()?;
        let locked_count = mutex_get_locked_count(this, mutex_op)?.to_u32()?;

        if kind == this.eval_libc("PTHREAD_MUTEX_NORMAL")? {
            if locked_count != 0 {
                mutex_set_locked_count(this, mutex_op, Scalar::from_u32(0))?;
                Ok(0)
            } else {
                throw_ub_format!(
                    "Attempted to unlock a PTHREAD_MUTEX_NORMAL mutex that was not locked"
                );
            }
        } else if kind == this.eval_libc("PTHREAD_MUTEX_ERRORCHECK")? {
            if locked_count != 0 {
                mutex_set_locked_count(this, mutex_op, Scalar::from_u32(0))?;
                Ok(0)
            } else {
                this.eval_libc_i32("EPERM")
            }
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
            this.eval_libc_i32("EINVAL")
        }
    }

    fn pthread_mutex_destroy(&mut self, mutex_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let mutex = this.read_scalar(mutex_op)?.not_undef()?;
        if this.is_null(mutex)? {
            return this.eval_libc_i32("EINVAL");
        }

        if mutex_get_locked_count(this, mutex_op)?.to_u32()? != 0 {
            return this.eval_libc_i32("EBUSY");
        }

        mutex_set_kind(this, mutex_op, ScalarMaybeUndef::Undef)?;
        mutex_set_locked_count(this, mutex_op, ScalarMaybeUndef::Undef)?;

        Ok(0)
    }

    fn pthread_rwlock_rdlock(&mut self, rwlock_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let rwlock = this.read_scalar(rwlock_op)?.not_undef()?;
        if this.is_null(rwlock)? {
            return this.eval_libc_i32("EINVAL");
        }

        let readers = rwlock_get_readers(this, rwlock_op)?.to_u32()?;
        let writers = rwlock_get_writers(this, rwlock_op)?.to_u32()?;
        if writers != 0 {
            throw_unsup_format!(
                "Deadlock due to read-locking a pthreads read-write lock while it is already write-locked"
            );
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

        let rwlock = this.read_scalar(rwlock_op)?.not_undef()?;
        if this.is_null(rwlock)? {
            return this.eval_libc_i32("EINVAL");
        }

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

        let rwlock = this.read_scalar(rwlock_op)?.not_undef()?;
        if this.is_null(rwlock)? {
            return this.eval_libc_i32("EINVAL");
        }

        let readers = rwlock_get_readers(this, rwlock_op)?.to_u32()?;
        let writers = rwlock_get_writers(this, rwlock_op)?.to_u32()?;
        if readers != 0 {
            throw_unsup_format!(
                "Deadlock due to write-locking a pthreads read-write lock while it is already read-locked"
            );
        } else if writers != 0 {
            throw_unsup_format!(
                "Deadlock due to write-locking a pthreads read-write lock while it is already write-locked"
            );
        } else {
            rwlock_set_writers(this, rwlock_op, Scalar::from_u32(1))?;
            Ok(0)
        }
    }

    fn pthread_rwlock_trywrlock(&mut self, rwlock_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let rwlock = this.read_scalar(rwlock_op)?.not_undef()?;
        if this.is_null(rwlock)? {
            return this.eval_libc_i32("EINVAL");
        }

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

        let rwlock = this.read_scalar(rwlock_op)?.not_undef()?;
        if this.is_null(rwlock)? {
            return this.eval_libc_i32("EINVAL");
        }

        let readers = rwlock_get_readers(this, rwlock_op)?.to_u32()?;
        let writers = rwlock_get_writers(this, rwlock_op)?.to_u32()?;
        if let Some(new_readers) = readers.checked_sub(1) {
            rwlock_set_readers(this, rwlock_op, Scalar::from_u32(new_readers))?;
            Ok(0)
        } else if writers != 0 {
            rwlock_set_writers(this, rwlock_op, Scalar::from_u32(0))?;
            Ok(0)
        } else {
            this.eval_libc_i32("EPERM")
        }
    }

    fn pthread_rwlock_destroy(&mut self, rwlock_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let rwlock = this.read_scalar(rwlock_op)?.not_undef()?;
        if this.is_null(rwlock)? {
            return this.eval_libc_i32("EINVAL");
        }

        if rwlock_get_readers(this, rwlock_op)?.to_u32()? != 0 {
            return this.eval_libc_i32("EBUSY");
        }
        if rwlock_get_writers(this, rwlock_op)?.to_u32()? != 0 {
            return this.eval_libc_i32("EBUSY");
        }

        rwlock_set_readers(this, rwlock_op, ScalarMaybeUndef::Undef)?;
        rwlock_set_writers(this, rwlock_op, ScalarMaybeUndef::Undef)?;

        Ok(0)
    }
}

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

// Our chosen memory layout: store an i32 in the first four bytes equal to the
// corresponding libc mutex kind constant (i.e. PTHREAD_MUTEX_NORMAL)

fn mutexattr_get_kind<'mir, 'tcx: 'mir>(
    ecx: &MiriEvalContext<'mir, 'tcx>,
    attr_op: OpTy<'tcx, Tag>,
) -> InterpResult<'tcx, ScalarMaybeUndef<Tag>> {
    // Ensure that the following read at an offset to the attr pointer is within bounds
    assert_ptr_target_min_size(ecx, attr_op, 4)?;
    let attr_place = ecx.deref_operand(attr_op)?;
    let i32_layout = ecx.layout_of(ecx.tcx.types.i32)?;
    let kind_place = attr_place.offset(Size::ZERO, MemPlaceMeta::None, i32_layout, ecx)?;
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
    let i32_layout = ecx.layout_of(ecx.tcx.types.i32)?;
    let kind_place = attr_place.offset(Size::ZERO, MemPlaceMeta::None, i32_layout, ecx)?;
    ecx.write_scalar(kind.into(), kind_place.into())
}

// pthread_mutex_t is between 24 and 48 bytes, depending on the platform.

// Our chosen memory layout:
// bytes 0-3: reserved for signature on macOS
// (need to avoid this because it is set by static initializer macros)
// bytes 4-7: count of how many times this mutex has been locked, as a u32
// bytes 12-15 or 16-19 (depending on platform): mutex kind, as an i32
// (the kind has to be at its offset for compatibility with static initializer macros)

static LIBC_MUTEX_KIND_OFFSET_CACHE: AtomicU64 = AtomicU64::new(0);

fn libc_mutex_kind_offset<'mir, 'tcx: 'mir>(
    ecx: &mut MiriEvalContext<'mir, 'tcx>,
) -> InterpResult<'tcx, u64> {
    // Check if this offset has already been found and memoized
    let cached_value = LIBC_MUTEX_KIND_OFFSET_CACHE.load(Ordering::Relaxed);
    if cached_value != 0 {
        return Ok(cached_value);
    }

    // This function infers the offset of the `kind` field of libc's pthread_mutex_t
    // C struct by examining the array inside libc::PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP.
    // At time of writing, it is always all zero bytes except for a one byte at one of
    // four positions, depending on the target OS's C struct layout and the endianness of the
    // target architecture. This offset will then be used in getters and setters below, so that
    // mutexes created from static initializers can be emulated with the correct behavior.
    let initializer_path = ["libc", "PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP"];
    let initializer_instance = ecx.resolve_path(&initializer_path);
    let initializer_cid = GlobalId { instance: initializer_instance, promoted: None };
    let initializer_const_val = ecx.const_eval_raw(initializer_cid)?;
    let array_mplacety = ecx.mplace_field(initializer_const_val, 0)?;
    let array_length = match array_mplacety.layout.fields {
        FieldsShape::Array { count, .. } => count,
        _ => bug!("Couldn't get array length from type {:?}", array_mplacety.layout.ty),
    };

    let kind_offset = if array_length < 20 {
        bug!("libc::PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP array was shorter than expected");
    } else if ecx.read_scalar(ecx.mplace_field(array_mplacety, 16)?.into())?.to_u8()? != 0 {
        // for little-endian architectures
        16
    } else if ecx.read_scalar(ecx.mplace_field(array_mplacety, 19)?.into())?.to_u8()? != 0 {
        // for big-endian architectures
        // (note that the i32 spans bytes 16 through 19, so the offset of the kind field is 16)
        16
    } else if ecx.read_scalar(ecx.mplace_field(array_mplacety, 12)?.into())?.to_u8()? != 0 {
        // for little-endian architectures
        12
    } else if ecx.read_scalar(ecx.mplace_field(array_mplacety, 15)?.into())?.to_u8()? != 0 {
        // for big-endian architectures
        // (note that the i32 spans bytes 12 through 15, so the offset of the kind field is 12)
        12
    } else {
        bug!("Couldn't determine offset of `kind` in pthread_mutex_t");
    };

    // Save offset to memoization cache for future calls
    LIBC_MUTEX_KIND_OFFSET_CACHE.store(kind_offset, Ordering::Relaxed);
    Ok(kind_offset)
}

fn mutex_get_locked_count<'mir, 'tcx: 'mir>(
    ecx: &MiriEvalContext<'mir, 'tcx>,
    mutex_op: OpTy<'tcx, Tag>,
) -> InterpResult<'tcx, ScalarMaybeUndef<Tag>> {
    // Ensure that the following read at an offset to the mutex pointer is within bounds
    assert_ptr_target_min_size(ecx, mutex_op, 20)?;
    let mutex_place = ecx.deref_operand(mutex_op)?;
    let u32_layout = ecx.layout_of(ecx.tcx.types.u32)?;
    let locked_count_place =
        mutex_place.offset(Size::from_bytes(4), MemPlaceMeta::None, u32_layout, ecx)?;
    ecx.read_scalar(locked_count_place.into())
}

fn mutex_set_locked_count<'mir, 'tcx: 'mir>(
    ecx: &mut MiriEvalContext<'mir, 'tcx>,
    mutex_op: OpTy<'tcx, Tag>,
    locked_count: impl Into<ScalarMaybeUndef<Tag>>,
) -> InterpResult<'tcx, ()> {
    // Ensure that the following write at an offset to the mutex pointer is within bounds
    assert_ptr_target_min_size(ecx, mutex_op, 20)?;
    let mutex_place = ecx.deref_operand(mutex_op)?;
    let u32_layout = ecx.layout_of(ecx.tcx.types.u32)?;
    let locked_count_place =
        mutex_place.offset(Size::from_bytes(4), MemPlaceMeta::None, u32_layout, ecx)?;
    ecx.write_scalar(locked_count.into(), locked_count_place.into())
}

fn mutex_get_kind<'mir, 'tcx: 'mir>(
    ecx: &mut MiriEvalContext<'mir, 'tcx>,
    mutex_op: OpTy<'tcx, Tag>,
) -> InterpResult<'tcx, ScalarMaybeUndef<Tag>> {
    // Ensure that the following read at an offset to the mutex pointer is within bounds
    assert_ptr_target_min_size(ecx, mutex_op, 20)?;
    let mutex_place = ecx.deref_operand(mutex_op)?;
    let i32_layout = ecx.layout_of(ecx.tcx.types.i32)?;
    let kind_place = mutex_place.offset(
        Size::from_bytes(libc_mutex_kind_offset(ecx)?),
        MemPlaceMeta::None,
        i32_layout,
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
    assert_ptr_target_min_size(ecx, mutex_op, 20)?;
    let mutex_place = ecx.deref_operand(mutex_op)?;
    let i32_layout = ecx.layout_of(ecx.tcx.types.i32)?;
    let kind_place = mutex_place.offset(
        Size::from_bytes(libc_mutex_kind_offset(ecx)?),
        MemPlaceMeta::None,
        i32_layout,
        ecx,
    )?;
    ecx.write_scalar(kind.into(), kind_place.into())
}

// pthread_rwlock_t is between 32 and 56 bytes, depending on the platform.

// Our chosen memory layout:
// bytes 0-3: reserved for signature on macOS
// (need to avoid this because it is set by static initializer macros)
// bytes 4-7: reader count, as a u32
// bytes 8-11: writer count, as a u32

fn rwlock_get_readers<'mir, 'tcx: 'mir>(
    ecx: &MiriEvalContext<'mir, 'tcx>,
    rwlock_op: OpTy<'tcx, Tag>,
) -> InterpResult<'tcx, ScalarMaybeUndef<Tag>> {
    // Ensure that the following read at an offset to the rwlock pointer is within bounds
    assert_ptr_target_min_size(ecx, rwlock_op, 12)?;
    let rwlock_place = ecx.deref_operand(rwlock_op)?;
    let u32_layout = ecx.layout_of(ecx.tcx.types.u32)?;
    let readers_place =
        rwlock_place.offset(Size::from_bytes(4), MemPlaceMeta::None, u32_layout, ecx)?;
    ecx.read_scalar(readers_place.into())
}

fn rwlock_set_readers<'mir, 'tcx: 'mir>(
    ecx: &mut MiriEvalContext<'mir, 'tcx>,
    rwlock_op: OpTy<'tcx, Tag>,
    readers: impl Into<ScalarMaybeUndef<Tag>>,
) -> InterpResult<'tcx, ()> {
    // Ensure that the following write at an offset to the rwlock pointer is within bounds
    assert_ptr_target_min_size(ecx, rwlock_op, 12)?;
    let rwlock_place = ecx.deref_operand(rwlock_op)?;
    let u32_layout = ecx.layout_of(ecx.tcx.types.u32)?;
    let readers_place =
        rwlock_place.offset(Size::from_bytes(4), MemPlaceMeta::None, u32_layout, ecx)?;
    ecx.write_scalar(readers.into(), readers_place.into())
}

fn rwlock_get_writers<'mir, 'tcx: 'mir>(
    ecx: &MiriEvalContext<'mir, 'tcx>,
    rwlock_op: OpTy<'tcx, Tag>,
) -> InterpResult<'tcx, ScalarMaybeUndef<Tag>> {
    // Ensure that the following read at an offset to the rwlock pointer is within bounds
    assert_ptr_target_min_size(ecx, rwlock_op, 12)?;
    let rwlock_place = ecx.deref_operand(rwlock_op)?;
    let u32_layout = ecx.layout_of(ecx.tcx.types.u32)?;
    let writers_place =
        rwlock_place.offset(Size::from_bytes(8), MemPlaceMeta::None, u32_layout, ecx)?;
    ecx.read_scalar(writers_place.into())
}

fn rwlock_set_writers<'mir, 'tcx: 'mir>(
    ecx: &mut MiriEvalContext<'mir, 'tcx>,
    rwlock_op: OpTy<'tcx, Tag>,
    writers: impl Into<ScalarMaybeUndef<Tag>>,
) -> InterpResult<'tcx, ()> {
    // Ensure that the following write at an offset to the rwlock pointer is within bounds
    assert_ptr_target_min_size(ecx, rwlock_op, 12)?;
    let rwlock_place = ecx.deref_operand(rwlock_op)?;
    let u32_layout = ecx.layout_of(ecx.tcx.types.u32)?;
    let writers_place =
        rwlock_place.offset(Size::from_bytes(8), MemPlaceMeta::None, u32_layout, ecx)?;
    ecx.write_scalar(writers.into(), writers_place.into())
}
