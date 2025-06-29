use rustc_abi::Size;

use crate::concurrency::sync::LAZY_INIT_COOKIE;
use crate::*;

/// Do a bytewise comparison of the two places, using relaxed atomic reads. This is used to check if
/// a synchronization primitive matches its static initializer value.
///
/// The reads happen in chunks of 4, so all racing accesses must also use that access size.
fn bytewise_equal_atomic_relaxed<'tcx>(
    ecx: &MiriInterpCx<'tcx>,
    left: &MPlaceTy<'tcx>,
    right: &MPlaceTy<'tcx>,
) -> InterpResult<'tcx, bool> {
    let size = left.layout.size;
    assert_eq!(size, right.layout.size);

    // We do this in chunks of 4, so that we are okay to race with (sufficiently aligned)
    // 4-byte atomic accesses.
    assert!(size.bytes().is_multiple_of(4));
    for i in 0..(size.bytes() / 4) {
        let offset = Size::from_bytes(i.strict_mul(4));
        let load = |place: &MPlaceTy<'tcx>| {
            let byte = place.offset(offset, ecx.machine.layouts.u32, ecx)?;
            ecx.read_scalar_atomic(&byte, AtomicReadOrd::Relaxed)?.to_u32()
        };
        let left = load(left)?;
        let right = load(right)?;
        if left != right {
            return interp_ok(false);
        }
    }

    interp_ok(true)
}

// # pthread_mutexattr_t
// We store some data directly inside the type, ignoring the platform layout:
// - kind: i32

#[inline]
fn mutexattr_kind_offset<'tcx>(ecx: &MiriInterpCx<'tcx>) -> InterpResult<'tcx, u64> {
    interp_ok(match &*ecx.tcx.sess.target.os {
        "linux" | "illumos" | "solaris" | "macos" | "freebsd" | "android" => 0,
        os => throw_unsup_format!("`pthread_mutexattr` is not supported on {os}"),
    })
}

fn mutexattr_get_kind<'tcx>(
    ecx: &MiriInterpCx<'tcx>,
    attr_ptr: &OpTy<'tcx>,
) -> InterpResult<'tcx, i32> {
    ecx.deref_pointer_and_read(
        attr_ptr,
        mutexattr_kind_offset(ecx)?,
        ecx.libc_ty_layout("pthread_mutexattr_t"),
        ecx.machine.layouts.i32,
    )?
    .to_i32()
}

fn mutexattr_set_kind<'tcx>(
    ecx: &mut MiriInterpCx<'tcx>,
    attr_ptr: &OpTy<'tcx>,
    kind: i32,
) -> InterpResult<'tcx, ()> {
    ecx.deref_pointer_and_write(
        attr_ptr,
        mutexattr_kind_offset(ecx)?,
        Scalar::from_i32(kind),
        ecx.libc_ty_layout("pthread_mutexattr_t"),
        ecx.machine.layouts.i32,
    )
}

/// To differentiate "the mutex kind has not been changed" from
/// "the mutex kind has been set to PTHREAD_MUTEX_DEFAULT and that is
/// equal to some other mutex kind", we make the default value of this
/// field *not* PTHREAD_MUTEX_DEFAULT but this special flag.
const PTHREAD_MUTEX_KIND_UNCHANGED: i32 = 0x8000000;

/// Translates the mutex kind from what is stored in pthread_mutexattr_t to our enum.
fn mutexattr_translate_kind<'tcx>(
    ecx: &MiriInterpCx<'tcx>,
    kind: i32,
) -> InterpResult<'tcx, MutexKind> {
    interp_ok(if kind == (ecx.eval_libc_i32("PTHREAD_MUTEX_NORMAL")) {
        MutexKind::Normal
    } else if kind == ecx.eval_libc_i32("PTHREAD_MUTEX_ERRORCHECK") {
        MutexKind::ErrorCheck
    } else if kind == ecx.eval_libc_i32("PTHREAD_MUTEX_RECURSIVE") {
        MutexKind::Recursive
    } else if kind == ecx.eval_libc_i32("PTHREAD_MUTEX_DEFAULT")
        || kind == PTHREAD_MUTEX_KIND_UNCHANGED
    {
        // We check this *last* since PTHREAD_MUTEX_DEFAULT may be numerically equal to one of the
        // others, and we want an explicit `mutexattr_settype` to work as expected.
        MutexKind::Default
    } else {
        throw_unsup_format!("unsupported type of mutex: {kind}");
    })
}

// # pthread_mutex_t
// We store some data directly inside the type, ignoring the platform layout:
// - init: u32

/// The mutex kind.
#[derive(Debug, Clone, Copy)]
enum MutexKind {
    Normal,
    Default,
    Recursive,
    ErrorCheck,
}

#[derive(Debug, Clone)]
struct PthreadMutex {
    mutex_ref: MutexRef,
    kind: MutexKind,
}

/// To ensure an initialized mutex that was moved somewhere else can be distinguished from
/// a statically initialized mutex that is used the first time, we pick some offset within
/// `pthread_mutex_t` and use it as an "initialized" flag.
fn mutex_init_offset<'tcx>(ecx: &MiriInterpCx<'tcx>) -> InterpResult<'tcx, Size> {
    let offset = match &*ecx.tcx.sess.target.os {
        "linux" | "illumos" | "solaris" | "freebsd" | "android" => 0,
        // macOS stores a signature in the first bytes, so we move to offset 4.
        "macos" => 4,
        os => throw_unsup_format!("`pthread_mutex` is not supported on {os}"),
    };
    let offset = Size::from_bytes(offset);

    // Sanity-check this against PTHREAD_MUTEX_INITIALIZER (but only once):
    // the `init` field must start out not equal to INIT_COOKIE.
    if !ecx.machine.pthread_mutex_sanity.replace(true) {
        let check_static_initializer = |name| {
            let static_initializer = ecx.eval_path(&["libc", name]);
            let init_field =
                static_initializer.offset(offset, ecx.machine.layouts.u32, ecx).unwrap();
            let init = ecx.read_scalar(&init_field).unwrap().to_u32().unwrap();
            assert_ne!(
                init, LAZY_INIT_COOKIE,
                "{name} is incompatible with our initialization cookie"
            );
        };

        check_static_initializer("PTHREAD_MUTEX_INITIALIZER");
        // Check non-standard initializers.
        match &*ecx.tcx.sess.target.os {
            "linux" => {
                check_static_initializer("PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP");
                check_static_initializer("PTHREAD_ERRORCHECK_MUTEX_INITIALIZER_NP");
                check_static_initializer("PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP");
            }
            "illumos" | "solaris" | "macos" | "freebsd" | "android" => {
                // No non-standard initializers.
            }
            os => throw_unsup_format!("`pthread_mutex` is not supported on {os}"),
        }
    }

    interp_ok(offset)
}

/// Eagerly create and initialize a new mutex.
fn mutex_create<'tcx>(
    ecx: &mut MiriInterpCx<'tcx>,
    mutex_ptr: &OpTy<'tcx>,
    kind: MutexKind,
) -> InterpResult<'tcx, PthreadMutex> {
    let mutex = ecx.deref_pointer_as(mutex_ptr, ecx.libc_ty_layout("pthread_mutex_t"))?;
    let id = ecx.machine.sync.mutex_create();
    let data = PthreadMutex { mutex_ref: id, kind };
    ecx.lazy_sync_init(&mutex, mutex_init_offset(ecx)?, data.clone())?;
    interp_ok(data)
}

/// Returns the mutex data stored at the address that `mutex_ptr` points to.
/// Will raise an error if the mutex has been moved since its first use.
fn mutex_get_data<'tcx, 'a>(
    ecx: &'a mut MiriInterpCx<'tcx>,
    mutex_ptr: &OpTy<'tcx>,
) -> InterpResult<'tcx, &'a PthreadMutex>
where
    'tcx: 'a,
{
    let mutex = ecx.deref_pointer_as(mutex_ptr, ecx.libc_ty_layout("pthread_mutex_t"))?;
    ecx.lazy_sync_get_data(
        &mutex,
        mutex_init_offset(ecx)?,
        || throw_ub_format!("`pthread_mutex_t` can't be moved after first use"),
        |ecx| {
            let kind = mutex_kind_from_static_initializer(ecx, &mutex)?;
            let id = ecx.machine.sync.mutex_create();
            interp_ok(PthreadMutex { mutex_ref: id, kind })
        },
    )
}

/// Returns the kind of a static initializer.
fn mutex_kind_from_static_initializer<'tcx>(
    ecx: &MiriInterpCx<'tcx>,
    mutex: &MPlaceTy<'tcx>,
) -> InterpResult<'tcx, MutexKind> {
    // All the static initializers recognized here *must* be checked in `mutex_init_offset`!
    let is_initializer =
        |name| bytewise_equal_atomic_relaxed(ecx, mutex, &ecx.eval_path(&["libc", name]));

    // PTHREAD_MUTEX_INITIALIZER is recognized on all targets.
    if is_initializer("PTHREAD_MUTEX_INITIALIZER")? {
        return interp_ok(MutexKind::Default);
    }
    // Support additional platform-specific initializers.
    match &*ecx.tcx.sess.target.os {
        "linux" =>
            if is_initializer("PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP")? {
                return interp_ok(MutexKind::Recursive);
            } else if is_initializer("PTHREAD_ERRORCHECK_MUTEX_INITIALIZER_NP")? {
                return interp_ok(MutexKind::ErrorCheck);
            },
        _ => {}
    }
    throw_unsup_format!("unsupported static initializer used for `pthread_mutex_t`");
}

// # pthread_rwlock_t
// We store some data directly inside the type, ignoring the platform layout:
// - init: u32

#[derive(Debug, Clone)]
struct PthreadRwLock {
    rwlock_ref: RwLockRef,
}

fn rwlock_init_offset<'tcx>(ecx: &MiriInterpCx<'tcx>) -> InterpResult<'tcx, Size> {
    let offset = match &*ecx.tcx.sess.target.os {
        "linux" | "illumos" | "solaris" | "freebsd" | "android" => 0,
        // macOS stores a signature in the first bytes, so we move to offset 4.
        "macos" => 4,
        os => throw_unsup_format!("`pthread_rwlock` is not supported on {os}"),
    };
    let offset = Size::from_bytes(offset);

    // Sanity-check this against PTHREAD_RWLOCK_INITIALIZER (but only once):
    // the `init` field must start out not equal to LAZY_INIT_COOKIE.
    if !ecx.machine.pthread_rwlock_sanity.replace(true) {
        let static_initializer = ecx.eval_path(&["libc", "PTHREAD_RWLOCK_INITIALIZER"]);
        let init_field = static_initializer.offset(offset, ecx.machine.layouts.u32, ecx).unwrap();
        let init = ecx.read_scalar(&init_field).unwrap().to_u32().unwrap();
        assert_ne!(
            init, LAZY_INIT_COOKIE,
            "PTHREAD_RWLOCK_INITIALIZER is incompatible with our initialization cookie"
        );
    }

    interp_ok(offset)
}

fn rwlock_get_data<'tcx, 'a>(
    ecx: &'a mut MiriInterpCx<'tcx>,
    rwlock_ptr: &OpTy<'tcx>,
) -> InterpResult<'tcx, &'a PthreadRwLock>
where
    'tcx: 'a,
{
    let rwlock = ecx.deref_pointer_as(rwlock_ptr, ecx.libc_ty_layout("pthread_rwlock_t"))?;
    ecx.lazy_sync_get_data(
        &rwlock,
        rwlock_init_offset(ecx)?,
        || throw_ub_format!("`pthread_rwlock_t` can't be moved after first use"),
        |ecx| {
            if !bytewise_equal_atomic_relaxed(
                ecx,
                &rwlock,
                &ecx.eval_path(&["libc", "PTHREAD_RWLOCK_INITIALIZER"]),
            )? {
                throw_unsup_format!("unsupported static initializer used for `pthread_rwlock_t`");
            }
            let rwlock_ref = ecx.machine.sync.rwlock_create();
            interp_ok(PthreadRwLock { rwlock_ref })
        },
    )
}

// # pthread_condattr_t
// We store some data directly inside the type, ignoring the platform layout:
// - clock: i32

#[inline]
fn condattr_clock_offset<'tcx>(ecx: &MiriInterpCx<'tcx>) -> InterpResult<'tcx, u64> {
    interp_ok(match &*ecx.tcx.sess.target.os {
        "linux" | "illumos" | "solaris" | "freebsd" | "android" => 0,
        // macOS does not have a clock attribute.
        os => throw_unsup_format!("`pthread_condattr` clock field is not supported on {os}"),
    })
}

fn condattr_get_clock_id<'tcx>(
    ecx: &MiriInterpCx<'tcx>,
    attr_ptr: &OpTy<'tcx>,
) -> InterpResult<'tcx, i32> {
    ecx.deref_pointer_and_read(
        attr_ptr,
        condattr_clock_offset(ecx)?,
        ecx.libc_ty_layout("pthread_condattr_t"),
        ecx.machine.layouts.i32,
    )?
    .to_i32()
}

fn condattr_set_clock_id<'tcx>(
    ecx: &mut MiriInterpCx<'tcx>,
    attr_ptr: &OpTy<'tcx>,
    clock_id: i32,
) -> InterpResult<'tcx, ()> {
    ecx.deref_pointer_and_write(
        attr_ptr,
        condattr_clock_offset(ecx)?,
        Scalar::from_i32(clock_id),
        ecx.libc_ty_layout("pthread_condattr_t"),
        ecx.machine.layouts.i32,
    )
}

/// Translates the clock from what is stored in pthread_condattr_t to our enum.
fn condattr_translate_clock_id<'tcx>(
    ecx: &MiriInterpCx<'tcx>,
    raw_id: i32,
) -> InterpResult<'tcx, ClockId> {
    interp_ok(if raw_id == ecx.eval_libc_i32("CLOCK_REALTIME") {
        ClockId::Realtime
    } else if raw_id == ecx.eval_libc_i32("CLOCK_MONOTONIC") {
        ClockId::Monotonic
    } else {
        throw_unsup_format!("unsupported clock id: {raw_id}");
    })
}

// # pthread_cond_t
// We store some data directly inside the type, ignoring the platform layout:
// - init: u32

fn cond_init_offset<'tcx>(ecx: &MiriInterpCx<'tcx>) -> InterpResult<'tcx, Size> {
    let offset = match &*ecx.tcx.sess.target.os {
        "linux" | "illumos" | "solaris" | "freebsd" | "android" => 0,
        // macOS stores a signature in the first bytes, so we move to offset 4.
        "macos" => 4,
        os => throw_unsup_format!("`pthread_cond` is not supported on {os}"),
    };
    let offset = Size::from_bytes(offset);

    // Sanity-check this against PTHREAD_COND_INITIALIZER (but only once):
    // the `init` field must start out not equal to LAZY_INIT_COOKIE.
    if !ecx.machine.pthread_condvar_sanity.replace(true) {
        let static_initializer = ecx.eval_path(&["libc", "PTHREAD_COND_INITIALIZER"]);
        let init_field = static_initializer.offset(offset, ecx.machine.layouts.u32, ecx).unwrap();
        let init = ecx.read_scalar(&init_field).unwrap().to_u32().unwrap();
        assert_ne!(
            init, LAZY_INIT_COOKIE,
            "PTHREAD_COND_INITIALIZER is incompatible with our initialization cookie"
        );
    }

    interp_ok(offset)
}

#[derive(Debug, Clone, Copy)]
enum ClockId {
    Realtime,
    Monotonic,
}

#[derive(Debug, Copy, Clone)]
struct PthreadCondvar {
    id: CondvarId,
    clock: ClockId,
}

fn cond_create<'tcx>(
    ecx: &mut MiriInterpCx<'tcx>,
    cond_ptr: &OpTy<'tcx>,
    clock: ClockId,
) -> InterpResult<'tcx, PthreadCondvar> {
    let cond = ecx.deref_pointer_as(cond_ptr, ecx.libc_ty_layout("pthread_cond_t"))?;
    let id = ecx.machine.sync.condvar_create();
    let data = PthreadCondvar { id, clock };
    ecx.lazy_sync_init(&cond, cond_init_offset(ecx)?, data)?;
    interp_ok(data)
}

fn cond_get_data<'tcx, 'a>(
    ecx: &'a mut MiriInterpCx<'tcx>,
    cond_ptr: &OpTy<'tcx>,
) -> InterpResult<'tcx, &'a PthreadCondvar>
where
    'tcx: 'a,
{
    let cond = ecx.deref_pointer_as(cond_ptr, ecx.libc_ty_layout("pthread_cond_t"))?;
    ecx.lazy_sync_get_data(
        &cond,
        cond_init_offset(ecx)?,
        || throw_ub_format!("`pthread_cond_t` can't be moved after first use"),
        |ecx| {
            if !bytewise_equal_atomic_relaxed(
                ecx,
                &cond,
                &ecx.eval_path(&["libc", "PTHREAD_COND_INITIALIZER"]),
            )? {
                throw_unsup_format!("unsupported static initializer used for `pthread_cond_t`");
            }
            // This used the static initializer. The clock there is always CLOCK_REALTIME.
            let id = ecx.machine.sync.condvar_create();
            interp_ok(PthreadCondvar { id, clock: ClockId::Realtime })
        },
    )
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn pthread_mutexattr_init(&mut self, attr_op: &OpTy<'tcx>) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_mut();

        mutexattr_set_kind(this, attr_op, PTHREAD_MUTEX_KIND_UNCHANGED)?;

        interp_ok(())
    }

    fn pthread_mutexattr_settype(
        &mut self,
        attr_op: &OpTy<'tcx>,
        kind_op: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let kind = this.read_scalar(kind_op)?.to_i32()?;
        if kind == this.eval_libc_i32("PTHREAD_MUTEX_NORMAL")
            || kind == this.eval_libc_i32("PTHREAD_MUTEX_DEFAULT")
            || kind == this.eval_libc_i32("PTHREAD_MUTEX_ERRORCHECK")
            || kind == this.eval_libc_i32("PTHREAD_MUTEX_RECURSIVE")
        {
            // Make sure we do not mix this up with the "unchanged" kind.
            assert_ne!(kind, PTHREAD_MUTEX_KIND_UNCHANGED);
            mutexattr_set_kind(this, attr_op, kind)?;
        } else {
            let einval = this.eval_libc_i32("EINVAL");
            return interp_ok(Scalar::from_i32(einval));
        }

        interp_ok(Scalar::from_i32(0))
    }

    fn pthread_mutexattr_destroy(&mut self, attr_op: &OpTy<'tcx>) -> InterpResult<'tcx, ()> {
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

        interp_ok(())
    }

    fn pthread_mutex_init(
        &mut self,
        mutex_op: &OpTy<'tcx>,
        attr_op: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_mut();

        let attr = this.read_pointer(attr_op)?;
        let kind = if this.ptr_is_null(attr)? {
            MutexKind::Default
        } else {
            mutexattr_translate_kind(this, mutexattr_get_kind(this, attr_op)?)?
        };

        mutex_create(this, mutex_op, kind)?;

        interp_ok(())
    }

    fn pthread_mutex_lock(
        &mut self,
        mutex_op: &OpTy<'tcx>,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let mutex = mutex_get_data(this, mutex_op)?.clone();

        let ret = if let Some(owner_thread) = mutex.mutex_ref.owner() {
            if owner_thread != this.active_thread() {
                this.mutex_enqueue_and_block(
                    mutex.mutex_ref,
                    Some((Scalar::from_i32(0), dest.clone())),
                );
                return interp_ok(());
            } else {
                // Trying to acquire the same mutex again.
                match mutex.kind {
                    MutexKind::Default =>
                        throw_ub_format!(
                            "trying to acquire default mutex already locked by the current thread"
                        ),
                    MutexKind::Normal => throw_machine_stop!(TerminationInfo::Deadlock),
                    MutexKind::ErrorCheck => this.eval_libc_i32("EDEADLK"),
                    MutexKind::Recursive => {
                        this.mutex_lock(&mutex.mutex_ref);
                        0
                    }
                }
            }
        } else {
            // The mutex is unlocked. Let's lock it.
            this.mutex_lock(&mutex.mutex_ref);
            0
        };
        this.write_scalar(Scalar::from_i32(ret), dest)?;
        interp_ok(())
    }

    fn pthread_mutex_trylock(&mut self, mutex_op: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let mutex = mutex_get_data(this, mutex_op)?.clone();

        interp_ok(Scalar::from_i32(if let Some(owner_thread) = mutex.mutex_ref.owner() {
            if owner_thread != this.active_thread() {
                this.eval_libc_i32("EBUSY")
            } else {
                match mutex.kind {
                    MutexKind::Default | MutexKind::Normal | MutexKind::ErrorCheck =>
                        this.eval_libc_i32("EBUSY"),
                    MutexKind::Recursive => {
                        this.mutex_lock(&mutex.mutex_ref);
                        0
                    }
                }
            }
        } else {
            // The mutex is unlocked. Let's lock it.
            this.mutex_lock(&mutex.mutex_ref);
            0
        }))
    }

    fn pthread_mutex_unlock(&mut self, mutex_op: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let mutex = mutex_get_data(this, mutex_op)?.clone();

        if let Some(_old_locked_count) = this.mutex_unlock(&mutex.mutex_ref)? {
            // The mutex was locked by the current thread.
            interp_ok(Scalar::from_i32(0))
        } else {
            // The mutex was locked by another thread or not locked at all. See
            // the “Unlock When Not Owner” column in
            // https://pubs.opengroup.org/onlinepubs/9699919799/functions/pthread_mutex_unlock.html.
            match mutex.kind {
                MutexKind::Default =>
                    throw_ub_format!(
                        "unlocked a default mutex that was not locked by the current thread"
                    ),
                MutexKind::Normal =>
                    throw_ub_format!(
                        "unlocked a PTHREAD_MUTEX_NORMAL mutex that was not locked by the current thread"
                    ),
                MutexKind::ErrorCheck | MutexKind::Recursive =>
                    interp_ok(Scalar::from_i32(this.eval_libc_i32("EPERM"))),
            }
        }
    }

    fn pthread_mutex_destroy(&mut self, mutex_op: &OpTy<'tcx>) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_mut();

        // Reading the field also has the side-effect that we detect double-`destroy`
        // since we make the field uninit below.
        let mutex = mutex_get_data(this, mutex_op)?.clone();

        if mutex.mutex_ref.owner().is_some() {
            throw_ub_format!("destroyed a locked mutex");
        }

        // This might lead to false positives, see comment in pthread_mutexattr_destroy
        this.write_uninit(
            &this.deref_pointer_as(mutex_op, this.libc_ty_layout("pthread_mutex_t"))?,
        )?;
        // FIXME: delete interpreter state associated with this mutex.

        interp_ok(())
    }

    fn pthread_rwlock_rdlock(
        &mut self,
        rwlock_op: &OpTy<'tcx>,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let rwlock = rwlock_get_data(this, rwlock_op)?.clone();

        if rwlock.rwlock_ref.is_write_locked() {
            this.rwlock_enqueue_and_block_reader(
                rwlock.rwlock_ref,
                Scalar::from_i32(0),
                dest.clone(),
            );
        } else {
            this.rwlock_reader_lock(&rwlock.rwlock_ref);
            this.write_null(dest)?;
        }

        interp_ok(())
    }

    fn pthread_rwlock_tryrdlock(&mut self, rwlock_op: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let rwlock = rwlock_get_data(this, rwlock_op)?.clone();

        if rwlock.rwlock_ref.is_write_locked() {
            interp_ok(Scalar::from_i32(this.eval_libc_i32("EBUSY")))
        } else {
            this.rwlock_reader_lock(&rwlock.rwlock_ref);
            interp_ok(Scalar::from_i32(0))
        }
    }

    fn pthread_rwlock_wrlock(
        &mut self,
        rwlock_op: &OpTy<'tcx>,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let rwlock = rwlock_get_data(this, rwlock_op)?.clone();

        if rwlock.rwlock_ref.is_locked() {
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
            this.rwlock_enqueue_and_block_writer(
                rwlock.rwlock_ref,
                Scalar::from_i32(0),
                dest.clone(),
            );
        } else {
            this.rwlock_writer_lock(&rwlock.rwlock_ref);
            this.write_null(dest)?;
        }

        interp_ok(())
    }

    fn pthread_rwlock_trywrlock(&mut self, rwlock_op: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let rwlock = rwlock_get_data(this, rwlock_op)?.clone();

        if rwlock.rwlock_ref.is_locked() {
            interp_ok(Scalar::from_i32(this.eval_libc_i32("EBUSY")))
        } else {
            this.rwlock_writer_lock(&rwlock.rwlock_ref);
            interp_ok(Scalar::from_i32(0))
        }
    }

    fn pthread_rwlock_unlock(&mut self, rwlock_op: &OpTy<'tcx>) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_mut();

        let rwlock = rwlock_get_data(this, rwlock_op)?.clone();

        if this.rwlock_reader_unlock(&rwlock.rwlock_ref)?
            || this.rwlock_writer_unlock(&rwlock.rwlock_ref)?
        {
            interp_ok(())
        } else {
            throw_ub_format!("unlocked an rwlock that was not locked by the active thread");
        }
    }

    fn pthread_rwlock_destroy(&mut self, rwlock_op: &OpTy<'tcx>) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_mut();

        // Reading the field also has the side-effect that we detect double-`destroy`
        // since we make the field uninit below.
        let rwlock = rwlock_get_data(this, rwlock_op)?.clone();

        if rwlock.rwlock_ref.is_locked() {
            throw_ub_format!("destroyed a locked rwlock");
        }

        // This might lead to false positives, see comment in pthread_mutexattr_destroy
        this.write_uninit(
            &this.deref_pointer_as(rwlock_op, this.libc_ty_layout("pthread_rwlock_t"))?,
        )?;
        // FIXME: delete interpreter state associated with this rwlock.

        interp_ok(())
    }

    fn pthread_condattr_init(&mut self, attr_op: &OpTy<'tcx>) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_mut();

        // no clock attribute on macOS
        if this.tcx.sess.target.os != "macos" {
            // The default value of the clock attribute shall refer to the system
            // clock.
            // https://pubs.opengroup.org/onlinepubs/9699919799/functions/pthread_condattr_setclock.html
            let default_clock_id = this.eval_libc_i32("CLOCK_REALTIME");
            condattr_set_clock_id(this, attr_op, default_clock_id)?;
        }

        interp_ok(())
    }

    fn pthread_condattr_setclock(
        &mut self,
        attr_op: &OpTy<'tcx>,
        clock_id_op: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let clock_id = this.read_scalar(clock_id_op)?.to_i32()?;
        if clock_id == this.eval_libc_i32("CLOCK_REALTIME")
            || clock_id == this.eval_libc_i32("CLOCK_MONOTONIC")
        {
            condattr_set_clock_id(this, attr_op, clock_id)?;
        } else {
            let einval = this.eval_libc_i32("EINVAL");
            return interp_ok(Scalar::from_i32(einval));
        }

        interp_ok(Scalar::from_i32(0))
    }

    fn pthread_condattr_getclock(
        &mut self,
        attr_op: &OpTy<'tcx>,
        clk_id_op: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_mut();

        let clock_id = condattr_get_clock_id(this, attr_op)?;
        this.write_scalar(
            Scalar::from_i32(clock_id),
            &this.deref_pointer_as(clk_id_op, this.libc_ty_layout("clockid_t"))?,
        )?;

        interp_ok(())
    }

    fn pthread_condattr_destroy(&mut self, attr_op: &OpTy<'tcx>) -> InterpResult<'tcx, ()> {
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

        interp_ok(())
    }

    fn pthread_cond_init(
        &mut self,
        cond_op: &OpTy<'tcx>,
        attr_op: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_mut();

        let attr = this.read_pointer(attr_op)?;
        // Default clock if `attr` is null, and on macOS where there is no clock attribute.
        let clock_id = if this.ptr_is_null(attr)? || this.tcx.sess.target.os == "macos" {
            this.eval_libc_i32("CLOCK_REALTIME")
        } else {
            condattr_get_clock_id(this, attr_op)?
        };
        let clock_id = condattr_translate_clock_id(this, clock_id)?;

        cond_create(this, cond_op, clock_id)?;

        interp_ok(())
    }

    fn pthread_cond_signal(&mut self, cond_op: &OpTy<'tcx>) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_mut();
        let id = cond_get_data(this, cond_op)?.id;
        this.condvar_signal(id)?;
        interp_ok(())
    }

    fn pthread_cond_broadcast(&mut self, cond_op: &OpTy<'tcx>) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_mut();
        let id = cond_get_data(this, cond_op)?.id;
        while this.condvar_signal(id)? {}
        interp_ok(())
    }

    fn pthread_cond_wait(
        &mut self,
        cond_op: &OpTy<'tcx>,
        mutex_op: &OpTy<'tcx>,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let data = *cond_get_data(this, cond_op)?;
        let mutex_ref = mutex_get_data(this, mutex_op)?.mutex_ref.clone();

        this.condvar_wait(
            data.id,
            mutex_ref,
            None, // no timeout
            Scalar::from_i32(0),
            Scalar::from_i32(0), // retval_timeout -- unused
            dest.clone(),
        )?;

        interp_ok(())
    }

    fn pthread_cond_timedwait(
        &mut self,
        cond_op: &OpTy<'tcx>,
        mutex_op: &OpTy<'tcx>,
        abstime_op: &OpTy<'tcx>,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let data = *cond_get_data(this, cond_op)?;
        let mutex_ref = mutex_get_data(this, mutex_op)?.mutex_ref.clone();

        // Extract the timeout.
        let duration = match this
            .read_timespec(&this.deref_pointer_as(abstime_op, this.libc_ty_layout("timespec"))?)?
        {
            Some(duration) => duration,
            None => {
                let einval = this.eval_libc("EINVAL");
                this.write_scalar(einval, dest)?;
                return interp_ok(());
            }
        };
        let timeout_clock = match data.clock {
            ClockId::Realtime => {
                this.check_no_isolation("`pthread_cond_timedwait` with `CLOCK_REALTIME`")?;
                TimeoutClock::RealTime
            }
            ClockId::Monotonic => TimeoutClock::Monotonic,
        };

        this.condvar_wait(
            data.id,
            mutex_ref,
            Some((timeout_clock, TimeoutAnchor::Absolute, duration)),
            Scalar::from_i32(0),
            this.eval_libc("ETIMEDOUT"), // retval_timeout
            dest.clone(),
        )?;

        interp_ok(())
    }

    fn pthread_cond_destroy(&mut self, cond_op: &OpTy<'tcx>) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_mut();

        // Reading the field also has the side-effect that we detect double-`destroy`
        // since we make the field uninit below.
        let id = cond_get_data(this, cond_op)?.id;
        if this.condvar_is_awaited(id) {
            throw_ub_format!("destroying an awaited conditional variable");
        }

        // This might lead to false positives, see comment in pthread_mutexattr_destroy
        this.write_uninit(&this.deref_pointer_as(cond_op, this.libc_ty_layout("pthread_cond_t"))?)?;
        // FIXME: delete interpreter state associated with this condvar.

        interp_ok(())
    }
}
