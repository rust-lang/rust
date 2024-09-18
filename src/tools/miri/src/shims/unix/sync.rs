use std::sync::atomic::{AtomicBool, Ordering};

use rustc_target::abi::Size;

use crate::*;

// pthread_mutexattr_t is either 4 or 8 bytes, depending on the platform.
// We ignore the platform layout and store our own fields:
// - kind: i32

#[inline]
fn mutexattr_kind_offset<'tcx>(ecx: &MiriInterpCx<'tcx>) -> InterpResult<'tcx, u64> {
    Ok(match &*ecx.tcx.sess.target.os {
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

/// The mutex kind.
#[derive(Debug, Clone, Copy)]
pub enum MutexKind {
    Normal,
    Default,
    Recursive,
    ErrorCheck,
}

#[derive(Debug)]
/// Additional data that we attach with each mutex instance.
pub struct AdditionalMutexData {
    /// The mutex kind, used by some mutex implementations like pthreads mutexes.
    pub kind: MutexKind,

    /// The address of the mutex.
    pub address: u64,
}

// pthread_mutex_t is between 4 and 48 bytes, depending on the platform.
// We ignore the platform layout and store our own fields:
// - id: u32

fn mutex_id_offset<'tcx>(ecx: &MiriInterpCx<'tcx>) -> InterpResult<'tcx, u64> {
    // When adding a new OS, make sure we also support all its static initializers in
    // `mutex_kind_from_static_initializer`!
    let offset = match &*ecx.tcx.sess.target.os {
        "linux" | "illumos" | "solaris" | "freebsd" | "android" => 0,
        // macOS stores a signature in the first bytes, so we have to move to offset 4.
        "macos" => 4,
        os => throw_unsup_format!("`pthread_mutex` is not supported on {os}"),
    };

    // Sanity-check this against PTHREAD_MUTEX_INITIALIZER (but only once):
    // the id must start out as 0.
    // FIXME on some platforms (e.g linux) there are more static initializers for
    // recursive or error checking mutexes. We should also add thme in this sanity check.
    static SANITY: AtomicBool = AtomicBool::new(false);
    if !SANITY.swap(true, Ordering::Relaxed) {
        let check_static_initializer = |name| {
            let static_initializer = ecx.eval_path(&["libc", name]);
            let id_field = static_initializer
                .offset(Size::from_bytes(offset), ecx.machine.layouts.u32, ecx)
                .unwrap();
            let id = ecx.read_scalar(&id_field).unwrap().to_u32().unwrap();
            assert_eq!(id, 0, "{name} is incompatible with our pthread_mutex layout: id is not 0");
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

    Ok(offset)
}

/// Eagerly create and initialize a new mutex.
fn mutex_create<'tcx>(
    ecx: &mut MiriInterpCx<'tcx>,
    mutex_ptr: &OpTy<'tcx>,
    kind: MutexKind,
) -> InterpResult<'tcx> {
    let mutex = ecx.deref_pointer(mutex_ptr)?;
    let address = mutex.ptr().addr().bytes();
    let data = Box::new(AdditionalMutexData { address, kind });
    ecx.mutex_create(&mutex, mutex_id_offset(ecx)?, Some(data))?;
    Ok(())
}

/// Returns the `MutexId` of the mutex stored at `mutex_op`.
///
/// `mutex_get_id` will also check if the mutex has been moved since its first use and
/// return an error if it has.
fn mutex_get_id<'tcx>(
    ecx: &mut MiriInterpCx<'tcx>,
    mutex_ptr: &OpTy<'tcx>,
) -> InterpResult<'tcx, MutexId> {
    let mutex = ecx.deref_pointer(mutex_ptr)?;
    let address = mutex.ptr().addr().bytes();

    let id = ecx.mutex_get_or_create_id(&mutex, mutex_id_offset(ecx)?, |ecx| {
        // This is called if a static initializer was used and the lock has not been assigned
        // an ID yet. We have to determine the mutex kind from the static initializer.
        let kind = mutex_kind_from_static_initializer(ecx, &mutex)?;

        Ok(Some(Box::new(AdditionalMutexData { kind, address })))
    })?;

    // Check that the mutex has not been moved since last use.
    let data = ecx
        .mutex_get_data::<AdditionalMutexData>(id)
        .expect("data should always exist for pthreads");
    if data.address != address {
        throw_ub_format!("pthread_mutex_t can't be moved after first use")
    }

    Ok(id)
}

/// Returns the kind of a static initializer.
fn mutex_kind_from_static_initializer<'tcx>(
    ecx: &MiriInterpCx<'tcx>,
    mutex: &MPlaceTy<'tcx>,
) -> InterpResult<'tcx, MutexKind> {
    Ok(match &*ecx.tcx.sess.target.os {
        // Only linux has static initializers other than PTHREAD_MUTEX_DEFAULT.
        "linux" => {
            let offset = if ecx.pointer_size().bytes() == 8 { 16 } else { 12 };
            let kind_place =
                mutex.offset(Size::from_bytes(offset), ecx.machine.layouts.i32, ecx)?;
            let kind = ecx.read_scalar(&kind_place)?.to_i32()?;
            // Here we give PTHREAD_MUTEX_DEFAULT priority so that
            // PTHREAD_MUTEX_INITIALIZER behaves like `pthread_mutex_init` with a NULL argument.
            if kind == ecx.eval_libc_i32("PTHREAD_MUTEX_DEFAULT") {
                MutexKind::Default
            } else {
                mutex_translate_kind(ecx, kind)?
            }
        }
        _ => MutexKind::Default,
    })
}

fn mutex_translate_kind<'tcx>(
    ecx: &MiriInterpCx<'tcx>,
    kind: i32,
) -> InterpResult<'tcx, MutexKind> {
    Ok(if kind == (ecx.eval_libc_i32("PTHREAD_MUTEX_NORMAL")) {
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

// pthread_rwlock_t is between 4 and 56 bytes, depending on the platform.
// We ignore the platform layout and store our own fields:
// - id: u32

#[derive(Debug)]
/// Additional data that we attach with each rwlock instance.
pub struct AdditionalRwLockData {
    /// The address of the rwlock.
    pub address: u64,
}

fn rwlock_id_offset<'tcx>(ecx: &MiriInterpCx<'tcx>) -> InterpResult<'tcx, u64> {
    let offset = match &*ecx.tcx.sess.target.os {
        "linux" | "illumos" | "solaris" | "freebsd" | "android" => 0,
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
    rwlock_ptr: &OpTy<'tcx>,
) -> InterpResult<'tcx, RwLockId> {
    let rwlock = ecx.deref_pointer(rwlock_ptr)?;
    let address = rwlock.ptr().addr().bytes();

    let id = ecx.rwlock_get_or_create_id(&rwlock, rwlock_id_offset(ecx)?, |_| {
        Ok(Some(Box::new(AdditionalRwLockData { address })))
    })?;

    // Check that the rwlock has not been moved since last use.
    let data = ecx
        .rwlock_get_data::<AdditionalRwLockData>(id)
        .expect("data should always exist for pthreads");
    if data.address != address {
        throw_ub_format!("pthread_rwlock_t can't be moved after first use")
    }

    Ok(id)
}

// pthread_condattr_t.
// We ignore the platform layout and store our own fields:
// - clock: i32

#[inline]
fn condattr_clock_offset<'tcx>(ecx: &MiriInterpCx<'tcx>) -> InterpResult<'tcx, u64> {
    Ok(match &*ecx.tcx.sess.target.os {
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

fn cond_translate_clock_id<'tcx>(
    ecx: &MiriInterpCx<'tcx>,
    raw_id: i32,
) -> InterpResult<'tcx, ClockId> {
    Ok(if raw_id == ecx.eval_libc_i32("CLOCK_REALTIME") {
        ClockId::Realtime
    } else if raw_id == ecx.eval_libc_i32("CLOCK_MONOTONIC") {
        ClockId::Monotonic
    } else {
        throw_unsup_format!("unsupported clock id: {raw_id}");
    })
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

// pthread_cond_t can be only 4 bytes in size, depending on the platform.
// We ignore the platform layout and store our own fields:
// - id: u32

fn cond_id_offset<'tcx>(ecx: &MiriInterpCx<'tcx>) -> InterpResult<'tcx, u64> {
    let offset = match &*ecx.tcx.sess.target.os {
        "linux" | "illumos" | "solaris" | "freebsd" | "android" => 0,
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

#[derive(Debug, Clone, Copy)]
enum ClockId {
    Realtime,
    Monotonic,
}

#[derive(Debug)]
/// Additional data that we attach with each cond instance.
struct AdditionalCondData {
    /// The address of the cond.
    address: u64,

    /// The clock id of the cond.
    clock_id: ClockId,
}

fn cond_get_id<'tcx>(
    ecx: &mut MiriInterpCx<'tcx>,
    cond_ptr: &OpTy<'tcx>,
) -> InterpResult<'tcx, CondvarId> {
    let cond = ecx.deref_pointer(cond_ptr)?;
    let address = cond.ptr().addr().bytes();
    let id = ecx.condvar_get_or_create_id(&cond, cond_id_offset(ecx)?, |_ecx| {
        // This used the static initializer. The clock there is always CLOCK_REALTIME.
        Ok(Some(Box::new(AdditionalCondData { address, clock_id: ClockId::Realtime })))
    })?;

    // Check that the mutex has not been moved since last use.
    let data = ecx
        .condvar_get_data::<AdditionalCondData>(id)
        .expect("data should always exist for pthreads");
    if data.address != address {
        throw_ub_format!("pthread_cond_t can't be moved after first use")
    }

    Ok(id)
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn pthread_mutexattr_init(&mut self, attr_op: &OpTy<'tcx>) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_mut();

        mutexattr_set_kind(this, attr_op, PTHREAD_MUTEX_KIND_UNCHANGED)?;

        Ok(())
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
            return Ok(Scalar::from_i32(einval));
        }

        Ok(Scalar::from_i32(0))
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

        Ok(())
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
            mutex_translate_kind(this, mutexattr_get_kind(this, attr_op)?)?
        };

        mutex_create(this, mutex_op, kind)?;

        Ok(())
    }

    fn pthread_mutex_lock(
        &mut self,
        mutex_op: &OpTy<'tcx>,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let id = mutex_get_id(this, mutex_op)?;
        let kind = this
            .mutex_get_data::<AdditionalMutexData>(id)
            .expect("data should always exist for pthread mutexes")
            .kind;

        let ret = if this.mutex_is_locked(id) {
            let owner_thread = this.mutex_get_owner(id);
            if owner_thread != this.active_thread() {
                this.mutex_enqueue_and_block(id, Some((Scalar::from_i32(0), dest.clone())));
                return Ok(());
            } else {
                // Trying to acquire the same mutex again.
                match kind {
                    MutexKind::Default =>
                        throw_ub_format!("trying to acquire already locked default mutex"),
                    MutexKind::Normal => throw_machine_stop!(TerminationInfo::Deadlock),
                    MutexKind::ErrorCheck => this.eval_libc_i32("EDEADLK"),
                    MutexKind::Recursive => {
                        this.mutex_lock(id);
                        0
                    }
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

    fn pthread_mutex_trylock(&mut self, mutex_op: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let id = mutex_get_id(this, mutex_op)?;
        let kind = this
            .mutex_get_data::<AdditionalMutexData>(id)
            .expect("data should always exist for pthread mutexes")
            .kind;

        Ok(Scalar::from_i32(if this.mutex_is_locked(id) {
            let owner_thread = this.mutex_get_owner(id);
            if owner_thread != this.active_thread() {
                this.eval_libc_i32("EBUSY")
            } else {
                match kind {
                    MutexKind::Default | MutexKind::Normal | MutexKind::ErrorCheck =>
                        this.eval_libc_i32("EBUSY"),
                    MutexKind::Recursive => {
                        this.mutex_lock(id);
                        0
                    }
                }
            }
        } else {
            // The mutex is unlocked. Let's lock it.
            this.mutex_lock(id);
            0
        }))
    }

    fn pthread_mutex_unlock(&mut self, mutex_op: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let id = mutex_get_id(this, mutex_op)?;
        let kind = this
            .mutex_get_data::<AdditionalMutexData>(id)
            .expect("data should always exist for pthread mutexes")
            .kind;

        if let Some(_old_locked_count) = this.mutex_unlock(id)? {
            // The mutex was locked by the current thread.
            Ok(Scalar::from_i32(0))
        } else {
            // The mutex was locked by another thread or not locked at all. See
            // the “Unlock When Not Owner” column in
            // https://pubs.opengroup.org/onlinepubs/9699919799/functions/pthread_mutex_unlock.html.
            match kind {
                MutexKind::Default =>
                    throw_ub_format!(
                        "unlocked a default mutex that was not locked by the current thread"
                    ),
                MutexKind::Normal =>
                    throw_ub_format!(
                        "unlocked a PTHREAD_MUTEX_NORMAL mutex that was not locked by the current thread"
                    ),
                MutexKind::ErrorCheck | MutexKind::Recursive =>
                    Ok(Scalar::from_i32(this.eval_libc_i32("EPERM"))),
            }
        }
    }

    fn pthread_mutex_destroy(&mut self, mutex_op: &OpTy<'tcx>) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_mut();

        // Reading the field also has the side-effect that we detect double-`destroy`
        // since we make the field unint below.
        let id = mutex_get_id(this, mutex_op)?;

        if this.mutex_is_locked(id) {
            throw_ub_format!("destroyed a locked mutex");
        }

        // This might lead to false positives, see comment in pthread_mutexattr_destroy
        this.write_uninit(
            &this.deref_pointer_as(mutex_op, this.libc_ty_layout("pthread_mutex_t"))?,
        )?;
        // FIXME: delete interpreter state associated with this mutex.

        Ok(())
    }

    fn pthread_rwlock_rdlock(
        &mut self,
        rwlock_op: &OpTy<'tcx>,
        dest: &MPlaceTy<'tcx>,
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

    fn pthread_rwlock_tryrdlock(&mut self, rwlock_op: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let id = rwlock_get_id(this, rwlock_op)?;

        if this.rwlock_is_write_locked(id) {
            Ok(Scalar::from_i32(this.eval_libc_i32("EBUSY")))
        } else {
            this.rwlock_reader_lock(id);
            Ok(Scalar::from_i32(0))
        }
    }

    fn pthread_rwlock_wrlock(
        &mut self,
        rwlock_op: &OpTy<'tcx>,
        dest: &MPlaceTy<'tcx>,
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

    fn pthread_rwlock_trywrlock(&mut self, rwlock_op: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let this = self.eval_context_mut();

        let id = rwlock_get_id(this, rwlock_op)?;

        if this.rwlock_is_locked(id) {
            Ok(Scalar::from_i32(this.eval_libc_i32("EBUSY")))
        } else {
            this.rwlock_writer_lock(id);
            Ok(Scalar::from_i32(0))
        }
    }

    fn pthread_rwlock_unlock(&mut self, rwlock_op: &OpTy<'tcx>) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_mut();

        let id = rwlock_get_id(this, rwlock_op)?;

        #[allow(clippy::if_same_then_else)]
        if this.rwlock_reader_unlock(id)? || this.rwlock_writer_unlock(id)? {
            Ok(())
        } else {
            throw_ub_format!("unlocked an rwlock that was not locked by the active thread");
        }
    }

    fn pthread_rwlock_destroy(&mut self, rwlock_op: &OpTy<'tcx>) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_mut();

        // Reading the field also has the side-effect that we detect double-`destroy`
        // since we make the field unint below.
        let id = rwlock_get_id(this, rwlock_op)?;

        if this.rwlock_is_locked(id) {
            throw_ub_format!("destroyed a locked rwlock");
        }

        // This might lead to false positives, see comment in pthread_mutexattr_destroy
        this.write_uninit(
            &this.deref_pointer_as(rwlock_op, this.libc_ty_layout("pthread_rwlock_t"))?,
        )?;
        // FIXME: delete interpreter state associated with this rwlock.

        Ok(())
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

        Ok(())
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
            return Ok(Scalar::from_i32(einval));
        }

        Ok(Scalar::from_i32(0))
    }

    fn pthread_condattr_getclock(
        &mut self,
        attr_op: &OpTy<'tcx>,
        clk_id_op: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_mut();

        let clock_id = condattr_get_clock_id(this, attr_op)?;
        this.write_scalar(Scalar::from_i32(clock_id), &this.deref_pointer(clk_id_op)?)?;

        Ok(())
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

        Ok(())
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
        let clock_id = cond_translate_clock_id(this, clock_id)?;

        let cond = this.deref_pointer(cond_op)?;
        let address = cond.ptr().addr().bytes();
        this.condvar_create(
            &cond,
            cond_id_offset(this)?,
            Some(Box::new(AdditionalCondData { address, clock_id })),
        )?;

        Ok(())
    }

    fn pthread_cond_signal(&mut self, cond_op: &OpTy<'tcx>) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_mut();
        let id = cond_get_id(this, cond_op)?;
        this.condvar_signal(id)?;
        Ok(())
    }

    fn pthread_cond_broadcast(&mut self, cond_op: &OpTy<'tcx>) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_mut();
        let id = cond_get_id(this, cond_op)?;
        while this.condvar_signal(id)? {}
        Ok(())
    }

    fn pthread_cond_wait(
        &mut self,
        cond_op: &OpTy<'tcx>,
        mutex_op: &OpTy<'tcx>,
        dest: &MPlaceTy<'tcx>,
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
        cond_op: &OpTy<'tcx>,
        mutex_op: &OpTy<'tcx>,
        abstime_op: &OpTy<'tcx>,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let id = cond_get_id(this, cond_op)?;
        let mutex_id = mutex_get_id(this, mutex_op)?;

        // Extract the timeout.
        let clock_id = this
            .condvar_get_data::<AdditionalCondData>(id)
            .expect("additional data should always be present for pthreads")
            .clock_id;
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
        let timeout_clock = match clock_id {
            ClockId::Realtime => {
                this.check_no_isolation("`pthread_cond_timedwait` with `CLOCK_REALTIME`")?;
                TimeoutClock::RealTime
            }
            ClockId::Monotonic => TimeoutClock::Monotonic,
        };

        this.condvar_wait(
            id,
            mutex_id,
            Some((timeout_clock, TimeoutAnchor::Absolute, duration)),
            Scalar::from_i32(0),
            this.eval_libc("ETIMEDOUT"), // retval_timeout
            dest.clone(),
        )?;

        Ok(())
    }

    fn pthread_cond_destroy(&mut self, cond_op: &OpTy<'tcx>) -> InterpResult<'tcx, ()> {
        let this = self.eval_context_mut();

        // Reading the field also has the side-effect that we detect double-`destroy`
        // since we make the field unint below.
        let id = cond_get_id(this, cond_op)?;
        if this.condvar_is_awaited(id) {
            throw_ub_format!("destroying an awaited conditional variable");
        }

        // This might lead to false positives, see comment in pthread_mutexattr_destroy
        this.write_uninit(&this.deref_pointer_as(cond_op, this.libc_ty_layout("pthread_cond_t"))?)?;
        // FIXME: delete interpreter state associated with this condvar.

        Ok(())
    }
}
