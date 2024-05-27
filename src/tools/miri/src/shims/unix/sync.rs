use std::sync::atomic::{AtomicBool, Ordering};
use std::time::SystemTime;

use rustc_target::abi::Size;

use crate::*;

// pthread_mutexattr_t is either 4 or 8 bytes, depending on the platform.
// We ignore the platform layout and store our own fields:
// - kind: i32

#[inline]
fn mutexattr_kind_offset<'tcx>(
    ecx: &MiriInterpCx<'tcx>,
) -> InterpResult<'tcx, u64> {
    Ok(match &*ecx.tcx.sess.target.os {
        "linux" | "illumos" | "solaris" | "macos" => 0,
        os => throw_unsup_format!("`pthread_mutexattr` is not supported on {os}"),
    })
}

fn mutexattr_get_kind<'tcx>(
    ecx: &MiriInterpCx<'tcx>,
    attr_op: &OpTy<'tcx, Provenance>,
) -> InterpResult<'tcx, i32> {
    ecx.deref_pointer_and_read(
        attr_op,
        mutexattr_kind_offset(ecx)?,
        ecx.libc_ty_layout("pthread_mutexattr_t"),
        ecx.machine.layouts.i32,
    )?
    .to_i32()
}

fn mutexattr_set_kind<'tcx>(
    ecx: &mut MiriInterpCx<'tcx>,
    attr_op: &OpTy<'tcx, Provenance>,
    kind: i32,
) -> InterpResult<'tcx, ()> {
    ecx.deref_pointer_and_write(
        attr_op,
        mutexattr_kind_offset(ecx)?,
        Scalar::from_i32(kind),
        ecx.libc_ty_layout("pthread_mutexattr_t"),
        ecx.machine.layouts.i32,
    )
}

/// A flag that allows to distinguish `PTHREAD_MUTEX_NORMAL` from
/// `PTHREAD_MUTEX_DEFAULT`. Since in `glibc` they have the same numeric values,
/// but different behaviour, we need a way to distinguish them. We do this by
/// setting this bit flag to the `PTHREAD_MUTEX_NORMAL` mutexes. See the comment
/// in `pthread_mutexattr_settype` function.
const PTHREAD_MUTEX_NORMAL_FLAG: i32 = 0x8000000;

fn is_mutex_kind_default<'tcx>(
    ecx: &MiriInterpCx<'tcx>,
    kind: i32,
) -> InterpResult<'tcx, bool> {
    Ok(kind == ecx.eval_libc_i32("PTHREAD_MUTEX_DEFAULT"))
}

fn is_mutex_kind_normal<'tcx>(
    ecx: &MiriInterpCx<'tcx>,
    kind: i32,
) -> InterpResult<'tcx, bool> {
    let mutex_normal_kind = ecx.eval_libc_i32("PTHREAD_MUTEX_NORMAL");
    Ok(kind == (mutex_normal_kind | PTHREAD_MUTEX_NORMAL_FLAG))
}

// pthread_mutex_t is between 24 and 48 bytes, depending on the platform.
// We ignore the platform layout and store our own fields:
// - id: u32
// - kind: i32

fn mutex_id_offset<'tcx>(ecx: &MiriInterpCx<'tcx>) -> InterpResult<'tcx, u64> {
    let offset = match &*ecx.tcx.sess.target.os {
        "linux" | "illumos" | "solaris" => 0,
        // macOS stores a signature in the first bytes, so we have to move to offset 4.
        "macos" => 4,
        os => throw_unsup_format!("`pthread_mutex` is not supported on {os}"),
    };

    // Sanity-check this against PTHREAD_MUTEX_INITIALIZER (but only once):
    // the id must start out as 0.
    static SANITY: AtomicBool = AtomicBool::new(false);
    if !SANITY.swap(true, Ordering::Relaxed) {
        let static_initializer = ecx.eval_path(&["libc", "PTHREAD_MUTEX_INITIALIZER"]);
        let id_field = static_initializer
            .offset(Size::from_bytes(offset), ecx.machine.layouts.u32, ecx)
            .unwrap();
        let id = ecx.read_scalar(&id_field).unwrap().to_u32().unwrap();
        assert_eq!(
            id, 0,
            "PTHREAD_MUTEX_INITIALIZER is incompatible with our pthread_mutex layout: id is not 0"
        );
    }

    Ok(offset)
}

fn mutex_kind_offset<'tcx>(ecx: &MiriInterpCx<'tcx>) -> u64 {
    // These offsets are picked for compatibility with Linux's static initializer
    // macros, e.g. PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP.)
    let offset = if ecx.pointer_size().bytes() == 8 { 16 } else { 12 };

    // Sanity-check this against PTHREAD_MUTEX_INITIALIZER (but only once):
    // the kind must start out as PTHREAD_MUTEX_DEFAULT.
    static SANITY: AtomicBool = AtomicBool::new(false);
    if !SANITY.swap(true, Ordering::Relaxed) {
        let static_initializer = ecx.eval_path(&["libc", "PTHREAD_MUTEX_INITIALIZER"]);
        let kind_field = static_initializer
            .offset(Size::from_bytes(mutex_kind_offset(ecx)), ecx.machine.layouts.i32, ecx)
            .unwrap();
        let kind = ecx.read_scalar(&kind_field).unwrap().to_i32().unwrap();
        assert_eq!(
            kind,
            ecx.eval_libc_i32("PTHREAD_MUTEX_DEFAULT"),
            "PTHREAD_MUTEX_INITIALIZER is incompatible with our pthread_mutex layout: kind is not PTHREAD_MUTEX_DEFAULT"
        );
    }

    offset
}

fn mutex_get_id<'tcx>(
    ecx: &mut MiriInterpCx<'tcx>,
    mutex_op: &OpTy<'tcx, Provenance>,
) -> InterpResult<'tcx, MutexId> {
    ecx.mutex_get_or_create_id(
        mutex_op,
        ecx.libc_ty_layout("pthread_mutex_t"),
        mutex_id_offset(ecx)?,
    )
}

fn mutex_reset_id<'tcx>(
    ecx: &mut MiriInterpCx<'tcx>,
    mutex_op: &OpTy<'tcx, Provenance>,
) -> InterpResult<'tcx, ()> {
    ecx.deref_pointer_and_write(
        mutex_op,
        mutex_id_offset(ecx)?,
        Scalar::from_u32(0),
        ecx.libc_ty_layout("pthread_mutex_t"),
        ecx.machine.layouts.u32,
    )
}

fn mutex_get_kind<'tcx>(
    ecx: &MiriInterpCx<'tcx>,
    mutex_op: &OpTy<'tcx, Provenance>,
) -> InterpResult<'tcx, i32> {
    ecx.deref_pointer_and_read(
        mutex_op,
        mutex_kind_offset(ecx),
        ecx.libc_ty_layout("pthread_mutex_t"),
        ecx.machine.layouts.i32,
    )?
    .to_i32()
}

fn mutex_set_kind<'tcx>(
    ecx: &mut MiriInterpCx<'tcx>,
    mutex_op: &OpTy<'tcx, Provenance>,
    kind: i32,
) -> InterpResult<'tcx, ()> {
    ecx.deref_pointer_and_write(
        mutex_op,
        mutex_kind_offset(ecx),
        Scalar::from_i32(kind),
        ecx.libc_ty_layout("pthread_mutex_t"),
        ecx.machine.layouts.i32,
    )
}

// pthread_rwlock_t is between 32 and 56 bytes, depending on the platform.
// We ignore the platform layout and store our own fields:
// - id: u32

fn rwlock_id_offset<'tcx>(ecx: &MiriInterpCx<'tcx>) -> InterpResult<'tcx, u64> {
    let offset = match &*ecx.tcx.sess.target.os {
        "linux" | "illumos" | "solaris" => 0,
        // macOS stores a signature in the first bytes, so we have to move to offset 4.
        "macos" => 4,
        os => throw_unsup_format!("`pthread_rwlock` is not supported on {os}"),
    };

    // Sanity-check this against PTHREAD_RWLOCK_INITIALIZER (but only once):
    // the id must start out as 0.
    static SANITY: AtomicBool = AtomicBool::new(false);
    if !SANITY.swap(true, Ordering::Relaxed) {
        let static_initializer = ecx.eval_path(&["libc", "PTHREAD_RWLOCK_INITIALIZER"]);
        let id_field = static_initializer
            .offset(Size::from_bytes(offset), ecx.machine.layouts.u32, ecx)
            .unwrap();
        let id = ecx.read_scalar(&id_field).unwrap().to_u32().unwrap();
        assert_eq!(
            id, 0,
            "PTHREAD_RWLOCK_INITIALIZER is incompatible with our pthread_rwlock layout: id is not 0"
        );
    }

    Ok(offset)
}

fn rwlock_get_id<'tcx>(
    ecx: &mut MiriInterpCx<'tcx>,
    rwlock_op: &OpTy<'tcx, Provenance>,
) -> InterpResult<'tcx, RwLockId> {
    ecx.rwlock_get_or_create_id(
        rwlock_op,
        ecx.libc_ty_layout("pthread_rwlock_t"),
        rwlock_id_offset(ecx)?,
    )
}

// pthread_condattr_t.
// We ignore the platform layout and store our own fields:
// - clock: i32

#[inline]
fn condattr_clock_offset<'tcx>(
    ecx: &MiriInterpCx<'tcx>,
) -> InterpResult<'tcx, u64> {
    Ok(match &*ecx.tcx.sess.target.os {
        "linux" | "illumos" | "solaris" => 0,
        // macOS does not have a clock attribute.
        os => throw_unsup_format!("`pthread_condattr` clock field is not supported on {os}"),
    })
}

fn condattr_get_clock_id<'tcx>(
    ecx: &MiriInterpCx<'tcx>,
    attr_op: &OpTy<'tcx, Provenance>,
) -> InterpResult<'tcx, i32> {
    ecx.deref_pointer_and_read(
        attr_op,
        condattr_clock_offset(ecx)?,
        ecx.libc_ty_layout("pthread_condattr_t"),
        ecx.machine.layouts.i32,
    )?
    .to_i32()
}

fn condattr_set_clock_id<'tcx>(
    ecx: &mut MiriInterpCx<'tcx>,
    attr_op: &OpTy<'tcx, Provenance>,
    clock_id: i32,
) -> InterpResult<'tcx, ()> {
    ecx.deref_pointer_and_write(
        attr_op,
        condattr_clock_offset(ecx)?,
        Scalar::from_i32(clock_id),
        ecx.libc_ty_layout("pthread_condattr_t"),
        ecx.machine.layouts.i32,
    )
}

// pthread_cond_t.
// We ignore the platform layout and store our own fields:
// - id: u32
// - clock: i32

fn cond_id_offset<'tcx>(ecx: &MiriInterpCx<'tcx>) -> InterpResult<'tcx, u64> {
    let offset = match &*ecx.tcx.sess.target.os {
        "linux" | "illumos" | "solaris" => 0,
        // macOS stores a signature in the first bytes, so we have to move to offset 4.
        "macos" => 4,
        os => throw_unsup_format!("`pthread_cond` is not supported on {os}"),
    };

    // Sanity-check this against PTHREAD_COND_INITIALIZER (but only once):
    // the id must start out as 0.
    static SANITY: AtomicBool = AtomicBool::new(false);
    if !SANITY.swap(true, Ordering::Relaxed) {
        let static_initializer = ecx.eval_path(&["libc", "PTHREAD_COND_INITIALIZER"]);
        let id_field = static_initializer
            .offset(Size::from_bytes(offset), ecx.machine.layouts.u32, ecx)
            .unwrap();
        let id = ecx.read_scalar(&id_field).unwrap().to_u32().unwrap();
        assert_eq!(
            id, 0,
            "PTHREAD_COND_INITIALIZER is incompatible with our pthread_cond layout: id is not 0"
        );
    }

    Ok(offset)
}

/// Determines whether this clock represents the real-time clock, CLOCK_REALTIME.
fn is_cond_clock_realtime<'tcx>(ecx: &MiriInterpCx<'tcx>, clock_id: i32) -> bool {
    // To ensure compatibility with PTHREAD_COND_INITIALIZER on all platforms,
    // we can't just compare with CLOCK_REALTIME: on Solarish, PTHREAD_COND_INITIALIZER
    // makes the clock 0 but CLOCK_REALTIME is 3.
    // However, we need to always be able to distinguish this from CLOCK_MONOTONIC.
    clock_id == ecx.eval_libc_i32("CLOCK_REALTIME")
        || (clock_id == 0 && clock_id != ecx.eval_libc_i32("CLOCK_MONOTONIC"))
}

fn cond_clock_offset<'tcx>(ecx: &MiriInterpCx<'tcx>) -> u64 {
    // macOS doesn't have a clock attribute, but to keep the code uniform we store
    // a clock ID in the pthread_cond_t anyway. There's enough space.
    let offset = 8;

    // Sanity-check this against PTHREAD_COND_INITIALIZER (but only once):
    // the clock must start out as CLOCK_REALTIME.
    static SANITY: AtomicBool = AtomicBool::new(false);
    if !SANITY.swap(true, Ordering::Relaxed) {
        let static_initializer = ecx.eval_path(&["libc", "PTHREAD_COND_INITIALIZER"]);
        let id_field = static_initializer
            .offset(Size::from_bytes(offset), ecx.machine.layouts.i32, ecx)
            .unwrap();
        let id = ecx.read_scalar(&id_field).unwrap().to_i32().unwrap();
        assert!(
            is_cond_clock_realtime(ecx, id),
            "PTHREAD_COND_INITIALIZER is incompatible with our pthread_cond layout: clock is not CLOCK_REALTIME"
        );
    }

    offset
}

fn cond_get_id<'tcx>(
    ecx: &mut MiriInterpCx<'tcx>,
    cond_op: &OpTy<'tcx, Provenance>,
) -> InterpResult<'tcx, CondvarId> {
    ecx.condvar_get_or_create_id(
        cond_op,
        ecx.libc_ty_layout("pthread_cond_t"),
        cond_id_offset(ecx)?,
    )
}

fn cond_reset_id<'tcx>(
    ecx: &mut MiriInterpCx<'tcx>,
    cond_op: &OpTy<'tcx, Provenance>,
) -> InterpResult<'tcx, ()> {
    ecx.deref_pointer_and_write(
        cond_op,
        cond_id_offset(ecx)?,
        Scalar::from_i32(0),
        ecx.libc_ty_layout("pthread_cond_t"),
        ecx.machine.layouts.u32,
    )
}

fn cond_get_clock_id<'tcx>(
    ecx: &MiriInterpCx<'tcx>,
    cond_op: &OpTy<'tcx, Provenance>,
) -> InterpResult<'tcx, i32> {
    ecx.deref_pointer_and_read(
        cond_op,
        cond_clock_offset(ecx),
        ecx.libc_ty_layout("pthread_cond_t"),
        ecx.machine.layouts.i32,
    )?
    .to_i32()
}

fn cond_set_clock_id<'tcx>(
    ecx: &mut MiriInterpCx<'tcx>,
    cond_op: &OpTy<'tcx, Provenance>,
    clock_id: i32,
) -> InterpResult<'tcx, ()> {
    ecx.deref_pointer_and_write(
        cond_op,
        cond_clock_offset(ecx),
        Scalar::from_i32(clock_id),
        ecx.libc_ty_layout("pthread_cond_t"),
        ecx.machine.layouts.i32,
    )
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn pthread_mutexattr_init(
        &mut self,
        attr_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let default_kind = this.eval_libc_i32("PTHREAD_MUTEX_DEFAULT");
        mutexattr_set_kind(this, attr_op, default_kind)?;

        Ok(0)
    }

    fn pthread_mutexattr_settype(
        &mut self,
        attr_op: &OpTy<'tcx, Provenance>,
        kind_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let kind = this.read_scalar(kind_op)?.to_i32()?;
        if kind == this.eval_libc_i32("PTHREAD_MUTEX_NORMAL") {
            // In `glibc` implementation, the numeric values of
            // `PTHREAD_MUTEX_NORMAL` and `PTHREAD_MUTEX_DEFAULT` are equal.
            // However, a mutex created by explicitly passing
            // `PTHREAD_MUTEX_NORMAL` type has in some cases different behaviour
            // from the default mutex for which the type was not explicitly
            // specified. For a more detailed discussion, please see
            // https://github.com/rust-lang/miri/issues/1419.
            //
            // To distinguish these two cases in already constructed mutexes, we
            // use the same trick as glibc: for the case when
            // `pthread_mutexattr_settype` is called explicitly, we set the
            // `PTHREAD_MUTEX_NORMAL_FLAG` flag.
            let normal_kind = kind | PTHREAD_MUTEX_NORMAL_FLAG;
            // Check that after setting the flag, the kind is distinguishable
            // from all other kinds.
            assert_ne!(normal_kind, this.eval_libc_i32("PTHREAD_MUTEX_DEFAULT"));
            assert_ne!(normal_kind, this.eval_libc_i32("PTHREAD_MUTEX_ERRORCHECK"));
            assert_ne!(normal_kind, this.eval_libc_i32("PTHREAD_MUTEX_RECURSIVE"));
            mutexattr_set_kind(this, attr_op, normal_kind)?;
        } else if kind == this.eval_libc_i32("PTHREAD_MUTEX_DEFAULT")
            || kind == this.eval_libc_i32("PTHREAD_MUTEX_ERRORCHECK")
            || kind == this.eval_libc_i32("PTHREAD_MUTEX_RECURSIVE")
        {
            mutexattr_set_kind(this, attr_op, kind)?;
        } else {
            let einval = this.eval_libc_i32("EINVAL");
            return Ok(einval);
        }

        Ok(0)
    }

    fn pthread_mutexattr_destroy(
        &mut self,
        attr_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        // Destroying an uninit pthread_mutexattr is UB, so check to make sure it's not uninit.
        mutexattr_get_kind(this, attr_op)?;

        // To catch double-destroys, we de-initialize the mutexattr.
        // This is technically not right and might lead to false positives. For example, the below
        // code is *likely* sound, even assuming uninit numbers are UB, but Miri complains.
        //
        // let mut x: MaybeUninit<libc::pthread_mutexattr_t> = MaybeUninit::zeroed();
        // libc::pthread_mutexattr_init(x.as_mut_ptr());
        // libc::pthread_mutexattr_destroy(x.as_mut_ptr());
        // x.assume_init();
        //
        // However, the way libstd uses the pthread APIs works in our favor here, so we can get away with this.
        // This can always be revisited to have some external state to catch double-destroys
        // but not complain about the above code. See https://github.com/rust-lang/miri/pull/1933
        this.write_uninit(
            &this.deref_pointer_as(attr_op, this.libc_ty_layout("pthread_mutexattr_t"))?,
        )?;

        Ok(0)
    }

    fn pthread_mutex_init(
        &mut self,
        mutex_op: &OpTy<'tcx, Provenance>,
        attr_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let attr = this.read_pointer(attr_op)?;
        let kind = if this.ptr_is_null(attr)? {
            this.eval_libc_i32("PTHREAD_MUTEX_DEFAULT")
        } else {
            mutexattr_get_kind(this, attr_op)?
        };

        // Write 0 to use the same code path as the static initializers.
        mutex_reset_id(this, mutex_op)?;

        mutex_set_kind(this, mutex_op, kind)?;

        Ok(0)
    }

    fn pthread_mutex_lock(
        &mut self,
        mutex_op: &OpTy<'tcx, Provenance>,
        dest: &MPlaceTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let kind = mutex_get_kind(this, mutex_op)?;
        let id = mutex_get_id(this, mutex_op)?;

        let ret = if this.mutex_is_locked(id) {
            let owner_thread = this.mutex_get_owner(id);
            if owner_thread != this.active_thread() {
                this.mutex_enqueue_and_block(id, Scalar::from_i32(0), dest.clone());
                return Ok(());
            } else {
                // Trying to acquire the same mutex again.
                if is_mutex_kind_default(this, kind)? {
                    throw_ub_format!("trying to acquire already locked default mutex");
                } else if is_mutex_kind_normal(this, kind)? {
                    throw_machine_stop!(TerminationInfo::Deadlock);
                } else if kind == this.eval_libc_i32("PTHREAD_MUTEX_ERRORCHECK") {
                    this.eval_libc_i32("EDEADLK")
                } else if kind == this.eval_libc_i32("PTHREAD_MUTEX_RECURSIVE") {
                    this.mutex_lock(id);
                    0
                } else {
                    throw_unsup_format!(
                        "called pthread_mutex_lock on an unsupported type of mutex"
                    );
                }
            }
        } else {
            // The mutex is unlocked. Let's lock it.
            this.mutex_lock(id);
            0
        };
        this.write_scalar(Scalar::from_i32(ret), dest)?;
        Ok(())
    }

    fn pthread_mutex_trylock(
        &mut self,
        mutex_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let kind = mutex_get_kind(this, mutex_op)?;
        let id = mutex_get_id(this, mutex_op)?;

        if this.mutex_is_locked(id) {
            let owner_thread = this.mutex_get_owner(id);
            if owner_thread != this.active_thread() {
                Ok(this.eval_libc_i32("EBUSY"))
            } else {
                if is_mutex_kind_default(this, kind)?
                    || is_mutex_kind_normal(this, kind)?
                    || kind == this.eval_libc_i32("PTHREAD_MUTEX_ERRORCHECK")
                {
                    Ok(this.eval_libc_i32("EBUSY"))
                } else if kind == this.eval_libc_i32("PTHREAD_MUTEX_RECURSIVE") {
                    this.mutex_lock(id);
                    Ok(0)
                } else {
                    throw_unsup_format!(
                        "called pthread_mutex_trylock on an unsupported type of mutex"
                    );
                }
            }
        } else {
            // The mutex is unlocked. Let's lock it.
            this.mutex_lock(id);
            Ok(0)
        }
    }

    fn pthread_mutex_unlock(
        &mut self,
        mutex_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let kind = mutex_get_kind(this, mutex_op)?;
        let id = mutex_get_id(this, mutex_op)?;

        if let Some(_old_locked_count) = this.mutex_unlock(id)? {
            // The mutex was locked by the current thread.
            Ok(0)
        } else {
            // The mutex was locked by another thread or not locked at all. See
            // the “Unlock When Not Owner” column in
            // https://pubs.opengroup.org/onlinepubs/9699919799/functions/pthread_mutex_unlock.html.
            if is_mutex_kind_default(this, kind)? {
                throw_ub_format!(
                    "unlocked a default mutex that was not locked by the current thread"
                );
            } else if is_mutex_kind_normal(this, kind)? {
                throw_ub_format!(
                    "unlocked a PTHREAD_MUTEX_NORMAL mutex that was not locked by the current thread"
                );
            } else if kind == this.eval_libc_i32("PTHREAD_MUTEX_ERRORCHECK")
                || kind == this.eval_libc_i32("PTHREAD_MUTEX_RECURSIVE")
            {
                Ok(this.eval_libc_i32("EPERM"))
            } else {
                throw_unsup_format!("called pthread_mutex_unlock on an unsupported type of mutex");
            }
        }
    }

    fn pthread_mutex_destroy(
        &mut self,
        mutex_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let id = mutex_get_id(this, mutex_op)?;

        if this.mutex_is_locked(id) {
            throw_ub_format!("destroyed a locked mutex");
        }

        // Destroying an uninit pthread_mutex is UB, so check to make sure it's not uninit.
        mutex_get_kind(this, mutex_op)?;
        mutex_get_id(this, mutex_op)?;

        // This might lead to false positives, see comment in pthread_mutexattr_destroy
        this.write_uninit(
            &this.deref_pointer_as(mutex_op, this.libc_ty_layout("pthread_mutex_t"))?,
        )?;
        // FIXME: delete interpreter state associated with this mutex.

        Ok(0)
    }

    fn pthread_rwlock_rdlock(
        &mut self,
        rwlock_op: &OpTy<'tcx, Provenance>,
        dest: &MPlaceTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let id = rwlock_get_id(this, rwlock_op)?;

        if this.rwlock_is_write_locked(id) {
            this.rwlock_enqueue_and_block_reader(id, Scalar::from_i32(0), dest.clone());
        } else {
            this.rwlock_reader_lock(id);
            this.write_null(dest)?;
        }

        Ok(())
    }

    fn pthread_rwlock_tryrdlock(
        &mut self,
        rwlock_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let id = rwlock_get_id(this, rwlock_op)?;

        if this.rwlock_is_write_locked(id) {
            Ok(this.eval_libc_i32("EBUSY"))
        } else {
            this.rwlock_reader_lock(id);
            Ok(0)
        }
    }

    fn pthread_rwlock_wrlock(
        &mut self,
        rwlock_op: &OpTy<'tcx, Provenance>,
        dest: &MPlaceTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let id = rwlock_get_id(this, rwlock_op)?;

        if this.rwlock_is_locked(id) {
            // Note: this will deadlock if the lock is already locked by this
            // thread in any way.
            //
            // Relevant documentation:
            // https://pubs.opengroup.org/onlinepubs/9699919799/functions/pthread_rwlock_wrlock.html
            // An in-depth discussion on this topic:
            // https://github.com/rust-lang/rust/issues/53127
            //
            // FIXME: Detect and report the deadlock proactively. (We currently
            // report the deadlock only when no thread can continue execution,
            // but we could detect that this lock is already locked and report
            // an error.)
            this.rwlock_enqueue_and_block_writer(id, Scalar::from_i32(0), dest.clone());
        } else {
            this.rwlock_writer_lock(id);
            this.write_null(dest)?;
        }

        Ok(())
    }

    fn pthread_rwlock_trywrlock(
        &mut self,
        rwlock_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let id = rwlock_get_id(this, rwlock_op)?;

        if this.rwlock_is_locked(id) {
            Ok(this.eval_libc_i32("EBUSY"))
        } else {
            this.rwlock_writer_lock(id);
            Ok(0)
        }
    }

    fn pthread_rwlock_unlock(
        &mut self,
        rwlock_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let id = rwlock_get_id(this, rwlock_op)?;

        #[allow(clippy::if_same_then_else)]
        if this.rwlock_reader_unlock(id)? {
            Ok(0)
        } else if this.rwlock_writer_unlock(id)? {
            Ok(0)
        } else {
            throw_ub_format!("unlocked an rwlock that was not locked by the active thread");
        }
    }

    fn pthread_rwlock_destroy(
        &mut self,
        rwlock_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let id = rwlock_get_id(this, rwlock_op)?;

        if this.rwlock_is_locked(id) {
            throw_ub_format!("destroyed a locked rwlock");
        }

        // Destroying an uninit pthread_rwlock is UB, so check to make sure it's not uninit.
        rwlock_get_id(this, rwlock_op)?;

        // This might lead to false positives, see comment in pthread_mutexattr_destroy
        this.write_uninit(
            &this.deref_pointer_as(rwlock_op, this.libc_ty_layout("pthread_rwlock_t"))?,
        )?;
        // FIXME: delete interpreter state associated with this rwlock.

        Ok(0)
    }

    fn pthread_condattr_init(
        &mut self,
        attr_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        // no clock attribute on macOS
        if this.tcx.sess.target.os != "macos" {
            // The default value of the clock attribute shall refer to the system
            // clock.
            // https://pubs.opengroup.org/onlinepubs/9699919799/functions/pthread_condattr_setclock.html
            let default_clock_id = this.eval_libc_i32("CLOCK_REALTIME");
            condattr_set_clock_id(this, attr_op, default_clock_id)?;
        }

        Ok(0)
    }

    fn pthread_condattr_setclock(
        &mut self,
        attr_op: &OpTy<'tcx, Provenance>,
        clock_id_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, Scalar<Provenance>> {
        let this = self.eval_context_mut();

        let clock_id = this.read_scalar(clock_id_op)?.to_i32()?;
        if clock_id == this.eval_libc_i32("CLOCK_REALTIME")
            || clock_id == this.eval_libc_i32("CLOCK_MONOTONIC")
        {
            condattr_set_clock_id(this, attr_op, clock_id)?;
        } else {
            let einval = this.eval_libc_i32("EINVAL");
            return Ok(Scalar::from_i32(einval));
        }

        Ok(Scalar::from_i32(0))
    }

    fn pthread_condattr_getclock(
        &mut self,
        attr_op: &OpTy<'tcx, Provenance>,
        clk_id_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, Scalar<Provenance>> {
        let this = self.eval_context_mut();

        let clock_id = condattr_get_clock_id(this, attr_op)?;
        this.write_scalar(Scalar::from_i32(clock_id), &this.deref_pointer(clk_id_op)?)?;

        Ok(Scalar::from_i32(0))
    }

    fn pthread_condattr_destroy(
        &mut self,
        attr_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        // Destroying an uninit pthread_condattr is UB, so check to make sure it's not uninit.
        // There's no clock attribute on macOS.
        if this.tcx.sess.target.os != "macos" {
            condattr_get_clock_id(this, attr_op)?;
        }

        // De-init the entire thing.
        // This might lead to false positives, see comment in pthread_mutexattr_destroy
        this.write_uninit(
            &this.deref_pointer_as(attr_op, this.libc_ty_layout("pthread_condattr_t"))?,
        )?;

        Ok(0)
    }

    fn pthread_cond_init(
        &mut self,
        cond_op: &OpTy<'tcx, Provenance>,
        attr_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let attr = this.read_pointer(attr_op)?;
        // Default clock if `attr` is null, and on macOS where there is no clock attribute.
        let clock_id = if this.ptr_is_null(attr)? || this.tcx.sess.target.os == "macos" {
            this.eval_libc_i32("CLOCK_REALTIME")
        } else {
            condattr_get_clock_id(this, attr_op)?
        };

        // Write 0 to use the same code path as the static initializers.
        cond_reset_id(this, cond_op)?;

        cond_set_clock_id(this, cond_op, clock_id)?;

        Ok(0)
    }

    fn pthread_cond_signal(&mut self, cond_op: &OpTy<'tcx, Provenance>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();
        let id = cond_get_id(this, cond_op)?;
        this.condvar_signal(id)?;
        Ok(0)
    }

    fn pthread_cond_broadcast(
        &mut self,
        cond_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();
        let id = cond_get_id(this, cond_op)?;
        while this.condvar_signal(id)? {}
        Ok(0)
    }

    fn pthread_cond_wait(
        &mut self,
        cond_op: &OpTy<'tcx, Provenance>,
        mutex_op: &OpTy<'tcx, Provenance>,
        dest: &MPlaceTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let id = cond_get_id(this, cond_op)?;
        let mutex_id = mutex_get_id(this, mutex_op)?;

        this.condvar_wait(
            id,
            mutex_id,
            None, // no timeout
            Scalar::from_i32(0),
            Scalar::from_i32(0), // retval_timeout -- unused
            dest.clone(),
        )?;

        Ok(())
    }

    fn pthread_cond_timedwait(
        &mut self,
        cond_op: &OpTy<'tcx, Provenance>,
        mutex_op: &OpTy<'tcx, Provenance>,
        abstime_op: &OpTy<'tcx, Provenance>,
        dest: &MPlaceTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let id = cond_get_id(this, cond_op)?;
        let mutex_id = mutex_get_id(this, mutex_op)?;

        // Extract the timeout.
        let clock_id = cond_get_clock_id(this, cond_op)?;
        let duration = match this
            .read_timespec(&this.deref_pointer_as(abstime_op, this.libc_ty_layout("timespec"))?)?
        {
            Some(duration) => duration,
            None => {
                let einval = this.eval_libc("EINVAL");
                this.write_scalar(einval, dest)?;
                return Ok(());
            }
        };
        let timeout_time = if is_cond_clock_realtime(this, clock_id) {
            this.check_no_isolation("`pthread_cond_timedwait` with `CLOCK_REALTIME`")?;
            Timeout::RealTime(SystemTime::UNIX_EPOCH.checked_add(duration).unwrap())
        } else if clock_id == this.eval_libc_i32("CLOCK_MONOTONIC") {
            Timeout::Monotonic(this.machine.clock.anchor().checked_add(duration).unwrap())
        } else {
            throw_unsup_format!("unsupported clock id: {}", clock_id);
        };

        this.condvar_wait(
            id,
            mutex_id,
            Some(timeout_time),
            Scalar::from_i32(0),
            this.eval_libc("ETIMEDOUT"), // retval_timeout
            dest.clone(),
        )?;

        Ok(())
    }

    fn pthread_cond_destroy(
        &mut self,
        cond_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        let id = cond_get_id(this, cond_op)?;
        if this.condvar_is_awaited(id) {
            throw_ub_format!("destroying an awaited conditional variable");
        }

        // Destroying an uninit pthread_cond is UB, so check to make sure it's not uninit.
        cond_get_id(this, cond_op)?;
        cond_get_clock_id(this, cond_op)?;

        // This might lead to false positives, see comment in pthread_mutexattr_destroy
        this.write_uninit(&this.deref_pointer_as(cond_op, this.libc_ty_layout("pthread_cond_t"))?)?;
        // FIXME: delete interpreter state associated with this condvar.

        Ok(0)
    }
}
