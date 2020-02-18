use rustc_middle::ty::{TyKind, TypeAndMut};
use rustc_target::abi::{LayoutOf, Size};

use crate::stacked_borrows::Tag;
use crate::*;

impl<'mir, 'tcx> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    // pthread_mutexattr_t is either 4 or 8 bytes, depending on the platform
    // memory layout: store an i32 in the first four bytes equal to the
    // corresponding libc mutex kind constant (i.e. PTHREAD_MUTEX_NORMAL)

    fn pthread_mutexattr_init(&mut self, attr_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        check_ptr_target_min_size(this, attr_op, 4)?;

        let attr = this.read_scalar(attr_op)?.not_undef()?;
        if this.is_null(attr)? {
            return this.eval_libc_i32("EINVAL");
        }

        let attr_place = this.deref_operand(attr_op)?;
        let i32_layout = this.layout_of(this.tcx.types.i32)?;
        let kind_place = attr_place.offset(Size::ZERO, MemPlaceMeta::None, i32_layout, this)?;
        let default_kind = this.eval_libc("PTHREAD_MUTEX_DEFAULT")?;
        this.write_scalar(default_kind, kind_place.into())?;

        Ok(0)
    }

    fn pthread_mutexattr_settype(
        &mut self,
        attr_op: OpTy<'tcx, Tag>,
        kind_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        check_ptr_target_min_size(this, attr_op, 4)?;

        let attr = this.read_scalar(attr_op)?.not_undef()?;
        if this.is_null(attr)? {
            return this.eval_libc_i32("EINVAL");
        }

        let kind = this.read_scalar(kind_op)?.not_undef()?;
        if kind == this.eval_libc("PTHREAD_MUTEX_NORMAL")? ||
                kind == this.eval_libc("PTHREAD_MUTEX_ERRORCHECK")? ||
                kind == this.eval_libc("PTHREAD_MUTEX_RECURSIVE")? {
            let attr_place = this.deref_operand(attr_op)?;
            let i32_layout = this.layout_of(this.tcx.types.i32)?;
            let kind_place = attr_place.offset(Size::ZERO, MemPlaceMeta::None, i32_layout, this)?;
            this.write_scalar(kind, kind_place.into())?;
        } else {
            let einval = this.eval_libc_i32("EINVAL")?;
            return Ok(einval);
        }

        Ok(0)
    }

    fn pthread_mutexattr_destroy(&mut self, attr_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        check_ptr_target_min_size(this, attr_op, 4)?;

        let attr = this.read_scalar(attr_op)?.not_undef()?;
        if this.is_null(attr)? {
            return this.eval_libc_i32("EINVAL");
        }

        let attr_place = this.deref_operand(attr_op)?;
        let i32_layout = this.layout_of(this.tcx.types.i32)?;
        let kind_place = attr_place.offset(Size::ZERO, MemPlaceMeta::None, i32_layout, this)?;
        this.write_scalar(ScalarMaybeUndef::Undef, kind_place.into())?;

        Ok(0)
    }

    // pthread_mutex_t is between 24 and 48 bytes, depending on the platform
    // memory layout:
    // bytes 0-3: reserved for signature on macOS
    // bytes 4-7: count of how many times this mutex has been locked, as a u32
    // bytes 12-15: mutex kind, as an i32
    // (the kind should be at this offset for compatibility with the static
    // initializer macro)

    fn pthread_mutex_init(
        &mut self,
        mutex_op: OpTy<'tcx, Tag>,
        attr_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        check_ptr_target_min_size(this, mutex_op, 16)?;
        check_ptr_target_min_size(this, attr_op, 4)?;

        let mutex = this.read_scalar(mutex_op)?.not_undef()?;
        if this.is_null(mutex)? {
            return this.eval_libc_i32("EINVAL");
        }
        let mutex_place = this.deref_operand(mutex_op)?;

        let i32_layout = this.layout_of(this.tcx.types.i32)?;

        let attr = this.read_scalar(attr_op)?.not_undef()?;
        let kind = if this.is_null(attr)? {
            this.eval_libc("PTHREAD_MUTEX_DEFAULT")?
        } else {
            let attr_place = this.deref_operand(attr_op)?;
            let attr_kind_place = attr_place.offset(Size::ZERO, MemPlaceMeta::None, i32_layout, this)?;
            this.read_scalar(attr_kind_place.into())?.not_undef()?
        };

        let u32_layout = this.layout_of(this.tcx.types.u32)?;
        let locked_count_place = mutex_place.offset(Size::from_bytes(4), MemPlaceMeta::None, u32_layout, this)?;
        this.write_scalar(Scalar::from_u32(0), locked_count_place.into())?;

        let mutex_kind_place = mutex_place.offset(Size::from_bytes(12), MemPlaceMeta::None, i32_layout, &*this.tcx)?;
        this.write_scalar(kind, mutex_kind_place.into())?;

        Ok(0)
    }

    fn pthread_mutex_lock(&mut self, mutex_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        check_ptr_target_min_size(this, mutex_op, 16)?;

        let mutex = this.read_scalar(mutex_op)?.not_undef()?;
        if this.is_null(mutex)? {
            return this.eval_libc_i32("EINVAL");
        }
        let mutex_place = this.deref_operand(mutex_op)?;

        let i32_layout = this.layout_of(this.tcx.types.i32)?;
        let kind_place = mutex_place.offset(Size::from_bytes(12), MemPlaceMeta::None, i32_layout, this)?;
        let kind = this.read_scalar(kind_place.into())?.not_undef()?;

        let u32_layout = this.layout_of(this.tcx.types.u32)?;
        let locked_count_place = mutex_place.offset(Size::from_bytes(4), MemPlaceMeta::None, u32_layout, this)?;
        let locked_count = this.read_scalar(locked_count_place.into())?.to_u32()?;

        if kind == this.eval_libc("PTHREAD_MUTEX_NORMAL")? {
            if locked_count == 0 {
                this.write_scalar(Scalar::from_u32(1), locked_count_place.into())?;
                Ok(0)
            } else {
                throw_unsup_format!("Deadlock due to locking a PTHREAD_MUTEX_NORMAL mutex twice");
            }
        } else if kind == this.eval_libc("PTHREAD_MUTEX_ERRORCHECK")? {
            if locked_count == 0 {
                this.write_scalar(Scalar::from_u32(1), locked_count_place.into())?;
                Ok(0)
            } else {
                this.eval_libc_i32("EDEADLK")
            }
        } else if kind == this.eval_libc("PTHREAD_MUTEX_RECURSIVE")? {
            this.write_scalar(Scalar::from_u32(locked_count + 1), locked_count_place.into())?;
            Ok(0)
        } else {
            this.eval_libc_i32("EINVAL")
        }
    }

    fn pthread_mutex_trylock(&mut self, mutex_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        check_ptr_target_min_size(this, mutex_op, 16)?;

        let mutex = this.read_scalar(mutex_op)?.not_undef()?;
        if this.is_null(mutex)? {
            return this.eval_libc_i32("EINVAL");
        }
        let mutex_place = this.deref_operand(mutex_op)?;

        let i32_layout = this.layout_of(this.tcx.types.i32)?;
        let kind_place = mutex_place.offset(Size::from_bytes(12), MemPlaceMeta::None, i32_layout, this)?;
        let kind = this.read_scalar(kind_place.into())?.not_undef()?;

        let u32_layout = this.layout_of(this.tcx.types.u32)?;
        let locked_count_place = mutex_place.offset(Size::from_bytes(4), MemPlaceMeta::None, u32_layout, this)?;
        let locked_count = this.read_scalar(locked_count_place.into())?.to_u32()?;

        if kind == this.eval_libc("PTHREAD_MUTEX_NORMAL")? ||
                kind == this.eval_libc("PTHREAD_MUTEX_ERRORCHECK")? {
            if locked_count == 0 {
                this.write_scalar(Scalar::from_u32(1), locked_count_place.into())?;
                Ok(0)
            } else {
                this.eval_libc_i32("EBUSY")
            }
        } else if kind == this.eval_libc("PTHREAD_MUTEX_RECURSIVE")? {
            this.write_scalar(Scalar::from_u32(locked_count + 1), locked_count_place.into())?;
            Ok(0)
        } else {
            this.eval_libc_i32("EINVAL")
        }
    }

    fn pthread_mutex_unlock(&mut self, mutex_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        check_ptr_target_min_size(this, mutex_op, 16)?;

        let mutex = this.read_scalar(mutex_op)?.not_undef()?;
        if this.is_null(mutex)? {
            return this.eval_libc_i32("EINVAL");
        }
        let mutex_place = this.deref_operand(mutex_op)?;

        let i32_layout = this.layout_of(this.tcx.types.i32)?;
        let kind_place = mutex_place.offset(Size::from_bytes(12), MemPlaceMeta::None, i32_layout, this)?;
        let kind = this.read_scalar(kind_place.into())?.not_undef()?;

        let u32_layout = this.layout_of(this.tcx.types.u32)?;
        let locked_count_place = mutex_place.offset(Size::from_bytes(4), MemPlaceMeta::None, u32_layout, this)?;
        let locked_count = this.read_scalar(locked_count_place.into())?.to_u32()?;

        if kind == this.eval_libc("PTHREAD_MUTEX_NORMAL")? {
            if locked_count == 1 {
                this.write_scalar(Scalar::from_u32(0), locked_count_place.into())?;
                Ok(0)
            } else {
                throw_ub_format!("Attempted to unlock a PTHREAD_MUTEX_NORMAL mutex that was not locked");
            }
        } else if kind == this.eval_libc("PTHREAD_MUTEX_ERRORCHECK")? {
            if locked_count == 1 {
                this.write_scalar(Scalar::from_u32(0), locked_count_place.into())?;
                Ok(0)
            } else {
                this.eval_libc_i32("EPERM")
            }
        } else if kind == this.eval_libc("PTHREAD_MUTEX_RECURSIVE")? {
            if locked_count > 0 {
                this.write_scalar(Scalar::from_u32(locked_count - 1), locked_count_place.into())?;
                Ok(0)
            } else {
                this.eval_libc_i32("EPERM")
            }
        } else {
            this.eval_libc_i32("EINVAL")
        }
    }

    fn pthread_mutex_destroy(&mut self, mutex_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        check_ptr_target_min_size(this, mutex_op, 16)?;

        let mutex = this.read_scalar(mutex_op)?.not_undef()?;
        if this.is_null(mutex)? {
            return this.eval_libc_i32("EINVAL");
        }
        let mutex_place = this.deref_operand(mutex_op)?;

        let u32_layout = this.layout_of(this.tcx.types.u32)?;
        let locked_count_place = mutex_place.offset(Size::from_bytes(4), MemPlaceMeta::None, u32_layout, this)?;
        if this.read_scalar(locked_count_place.into())?.to_u32()? != 0 {
            return this.eval_libc_i32("EBUSY");
        }

        let i32_layout = this.layout_of(this.tcx.types.i32)?;
        let kind_place = mutex_place.offset(Size::from_bytes(12), MemPlaceMeta::None, i32_layout, this)?;
        this.write_scalar(ScalarMaybeUndef::Undef, kind_place.into())?;
        this.write_scalar(ScalarMaybeUndef::Undef, locked_count_place.into())?;

        Ok(0)
    }

    // pthread_rwlock_t is between 32 and 56 bytes, depending on the platform
    // memory layout:
    // bytes 0-3: reserved for signature on macOS
    // bytes 4-7: reader count, as a u32
    // bytes 8-11: writer count, as a u32

    fn pthread_rwlock_rdlock(&mut self, rwlock_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        check_ptr_target_min_size(this, rwlock_op, 12)?;

        let rwlock = this.read_scalar(rwlock_op)?.not_undef()?;
        if this.is_null(rwlock)? {
            return this.eval_libc_i32("EINVAL");
        }
        let rwlock_place = this.deref_operand(rwlock_op)?;

        let u32_layout = this.layout_of(this.tcx.types.u32)?;
        let readers_place = rwlock_place.offset(Size::from_bytes(4), MemPlaceMeta::None, u32_layout, this)?;
        let writers_place = rwlock_place.offset(Size::from_bytes(8), MemPlaceMeta::None, u32_layout, this)?;
        let readers = this.read_scalar(readers_place.into())?.to_u32()?;
        let writers = this.read_scalar(writers_place.into())?.to_u32()?;
        if writers != 0 {
            throw_unsup_format!("Deadlock due to read-locking a pthreads read-write lock while it is already write-locked");
        } else {
            this.write_scalar(Scalar::from_u32(readers + 1), readers_place.into())?;
            Ok(0)
        }
    }

    fn pthread_rwlock_tryrdlock(&mut self, rwlock_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        check_ptr_target_min_size(this, rwlock_op, 12)?;

        let rwlock = this.read_scalar(rwlock_op)?.not_undef()?;
        if this.is_null(rwlock)? {
            return this.eval_libc_i32("EINVAL");
        }
        let rwlock_place = this.deref_operand(rwlock_op)?;

        let u32_layout = this.layout_of(this.tcx.types.u32)?;
        let readers_place = rwlock_place.offset(Size::from_bytes(4), MemPlaceMeta::None, u32_layout, this)?;
        let writers_place = rwlock_place.offset(Size::from_bytes(8), MemPlaceMeta::None, u32_layout, this)?;
        let readers = this.read_scalar(readers_place.into())?.to_u32()?;
        let writers = this.read_scalar(writers_place.into())?.to_u32()?;
        if writers != 0 {
            this.eval_libc_i32("EBUSY")
        } else {
            this.write_scalar(Scalar::from_u32(readers + 1), readers_place.into())?;
            Ok(0)
        }
    }

    fn pthread_rwlock_wrlock(&mut self, rwlock_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        check_ptr_target_min_size(this, rwlock_op, 12)?;

        let rwlock = this.read_scalar(rwlock_op)?.not_undef()?;
        if this.is_null(rwlock)? {
            return this.eval_libc_i32("EINVAL");
        }
        let rwlock_place = this.deref_operand(rwlock_op)?;

        let u32_layout = this.layout_of(this.tcx.types.u32)?;
        let readers_place = rwlock_place.offset(Size::from_bytes(4), MemPlaceMeta::None, u32_layout, this)?;
        let writers_place = rwlock_place.offset(Size::from_bytes(8), MemPlaceMeta::None, u32_layout, this)?;
        let readers = this.read_scalar(readers_place.into())?.to_u32()?;
        let writers = this.read_scalar(writers_place.into())?.to_u32()?;
        if readers != 0 {
            throw_unsup_format!("Deadlock due to write-locking a pthreads read-write lock while it is already read-locked");
        } else if writers != 0 {
            throw_unsup_format!("Deadlock due to write-locking a pthreads read-write lock while it is already write-locked");
        } else {
            this.write_scalar(Scalar::from_u32(1), writers_place.into())?;
            Ok(0)
        }
    }

    fn pthread_rwlock_trywrlock(&mut self, rwlock_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        check_ptr_target_min_size(this, rwlock_op, 12)?;

        let rwlock = this.read_scalar(rwlock_op)?.not_undef()?;
        if this.is_null(rwlock)? {
            return this.eval_libc_i32("EINVAL");
        }
        let rwlock_place = this.deref_operand(rwlock_op)?;

        let u32_layout = this.layout_of(this.tcx.types.u32)?;
        let readers_place = rwlock_place.offset(Size::from_bytes(4), MemPlaceMeta::None, u32_layout, this)?;
        let writers_place = rwlock_place.offset(Size::from_bytes(8), MemPlaceMeta::None, u32_layout, this)?;
        let readers = this.read_scalar(readers_place.into())?.to_u32()?;
        let writers = this.read_scalar(writers_place.into())?.to_u32()?;
        if readers != 0 || writers != 0 {
            this.eval_libc_i32("EBUSY")
        } else {
            this.write_scalar(Scalar::from_u32(1), writers_place.into())?;
            Ok(0)
        }
    }

    fn pthread_rwlock_unlock(&mut self, rwlock_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        check_ptr_target_min_size(this, rwlock_op, 12)?;

        let rwlock = this.read_scalar(rwlock_op)?.not_undef()?;
        if this.is_null(rwlock)? {
            return this.eval_libc_i32("EINVAL");
        }
        let rwlock_place = this.deref_operand(rwlock_op)?;

        let u32_layout = this.layout_of(this.tcx.types.u32)?;
        let readers_place = rwlock_place.offset(Size::from_bytes(4), MemPlaceMeta::None, u32_layout, this)?;
        let writers_place = rwlock_place.offset(Size::from_bytes(8), MemPlaceMeta::None, u32_layout, this)?;
        let readers = this.read_scalar(readers_place.into())?.to_u32()?;
        let writers = this.read_scalar(writers_place.into())?.to_u32()?;
        if readers != 0 {
            this.write_scalar(Scalar::from_u32(readers - 1), readers_place.into())?;
            Ok(0)
        } else if writers != 0 {
            this.write_scalar(Scalar::from_u32(0), writers_place.into())?;
            Ok(0)
        } else {
            this.eval_libc_i32("EPERM")
        }
    }

    fn pthread_rwlock_destroy(&mut self, rwlock_op: OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        check_ptr_target_min_size(this, rwlock_op, 12)?;

        let rwlock = this.read_scalar(rwlock_op)?.not_undef()?;
        if this.is_null(rwlock)? {
            return this.eval_libc_i32("EINVAL");
        }
        let rwlock_place = this.deref_operand(rwlock_op)?;

        let u32_layout = this.layout_of(this.tcx.types.u32)?;
        let readers_place = rwlock_place.offset(Size::from_bytes(4), MemPlaceMeta::None, u32_layout, this)?;
        if this.read_scalar(readers_place.into())?.to_u32()? != 0 {
            return this.eval_libc_i32("EBUSY");
        }
        let writers_place = rwlock_place.offset(Size::from_bytes(8), MemPlaceMeta::None, u32_layout, this)?;
        if this.read_scalar(writers_place.into())?.to_u32()? != 0 {
            return this.eval_libc_i32("EBUSY");
        }

        this.write_scalar(ScalarMaybeUndef::Undef, readers_place.into())?;
        this.write_scalar(ScalarMaybeUndef::Undef, writers_place.into())?;

        Ok(0)
    }
}

fn check_ptr_target_min_size<'mir, 'tcx: 'mir>(ecx: &MiriEvalContext<'mir, 'tcx>, operand: OpTy<'tcx, Tag>, min_size: u64) -> InterpResult<'tcx, ()> {
    let target_ty = match operand.layout.ty.kind {
        TyKind::RawPtr(TypeAndMut{ ty, mutbl: _ }) => ty,
        _ => panic!("Argument to pthread function was not a raw pointer"),
    };
    let target_layout = ecx.layout_of(target_ty)?;
    assert!(target_layout.size.bytes() >= min_size);
    Ok(())
}
