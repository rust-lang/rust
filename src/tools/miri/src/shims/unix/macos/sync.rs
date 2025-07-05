//! Contains macOS-specific synchronization functions.
//!
//! For `os_unfair_lock`, see the documentation
//! <https://developer.apple.com/documentation/os/synchronization?language=objc>
//! and in case of underspecification its implementation
//! <https://github.com/apple-oss-distributions/libplatform/blob/a00a4cc36da2110578bcf3b8eeeeb93dcc7f4e11/src/os/lock.c#L645>.
//!
//! Note that we don't emulate every edge-case behaviour of the locks. Notably,
//! we don't abort when locking a lock owned by a thread that has already exited
//! and we do not detect copying of the lock, but macOS doesn't guarantee anything
//! in that case either.

use std::cell::Cell;
use std::time::Duration;

use rustc_abi::Size;

use crate::concurrency::sync::FutexRef;
use crate::*;

#[derive(Clone)]
enum MacOsUnfairLock {
    Poisoned,
    Active { mutex_ref: MutexRef },
}

pub enum MacOsFutexTimeout<'a, 'tcx> {
    None,
    Relative { clock_op: &'a OpTy<'tcx>, timeout_op: &'a OpTy<'tcx> },
    Absolute { clock_op: &'a OpTy<'tcx>, timeout_op: &'a OpTy<'tcx> },
}

/// Metadata for a macOS futex.
///
/// Since macOS 11.0, Apple has exposed the previously private futex API consisting
/// of `os_sync_wait_on_address` (and friends) and `os_sync_wake_by_address_{any, all}`.
/// These work with different value sizes and flags, which are validated to be consistent.
/// This structure keeps track of both the futex queue and these values.
struct MacOsFutex {
    futex: FutexRef,
    /// The size in bytes of the atomic primitive underlying this futex.
    size: Cell<u64>,
    /// Whether the futex is shared across process boundaries.
    shared: Cell<bool>,
}

impl<'tcx> EvalContextExtPriv<'tcx> for crate::MiriInterpCx<'tcx> {}
trait EvalContextExtPriv<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn os_unfair_lock_get_data<'a>(
        &'a mut self,
        lock_ptr: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, &'a MacOsUnfairLock>
    where
        'tcx: 'a,
    {
        let this = self.eval_context_mut();
        let lock = this.deref_pointer_as(lock_ptr, this.libc_ty_layout("os_unfair_lock_s"))?;
        this.lazy_sync_get_data(
            &lock,
            Size::ZERO, // offset for init tracking
            || {
                // If we get here, due to how we reset things to zero in `os_unfair_lock_unlock`,
                // this means the lock was moved while locked. This can happen with a `std` lock,
                // but then any future attempt to unlock will just deadlock. In practice, terrible
                // things can probably happen if you swap two locked locks, since they'd wake up
                // from the wrong queue... we just won't catch all UB of this library API then (we
                // would need to store some unique identifer in-memory for this, instead of a static
                // LAZY_INIT_COOKIE). This can't be hit via `std::sync::Mutex`.
                interp_ok(MacOsUnfairLock::Poisoned)
            },
            |_| interp_ok(MacOsUnfairLock::Active { mutex_ref: MutexRef::new() }),
        )
    }
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Implements [`os_sync_wait_on_address`], [`os_sync_wait_on_address_with_deadline`]
    /// and [`os_sync_wait_on_address_with_timeout`].
    ///
    /// [`os_sync_wait_on_address`]: https://developer.apple.com/documentation/os/os_sync_wait_on_address?language=objc
    /// [`os_sync_wait_on_address_with_deadline`]: https://developer.apple.com/documentation/os/os_sync_wait_on_address_with_deadline?language=objc
    /// [`os_sync_wait_on_address_with_timeout`]: https://developer.apple.com/documentation/os/os_sync_wait_on_address_with_timeout?language=objc
    fn os_sync_wait_on_address(
        &mut self,
        addr_op: &OpTy<'tcx>,
        value_op: &OpTy<'tcx>,
        size_op: &OpTy<'tcx>,
        flags_op: &OpTy<'tcx>,
        timeout: MacOsFutexTimeout<'_, 'tcx>,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let none = this.eval_libc_u32("OS_SYNC_WAIT_ON_ADDRESS_NONE");
        let shared = this.eval_libc_u32("OS_SYNC_WAIT_ON_ADDRESS_SHARED");
        let absolute_clock = this.eval_libc_u32("OS_CLOCK_MACH_ABSOLUTE_TIME");

        let ptr = this.read_pointer(addr_op)?;
        let value = this.read_scalar(value_op)?.to_u64()?;
        let size = this.read_target_usize(size_op)?;
        let flags = this.read_scalar(flags_op)?.to_u32()?;

        let clock_timeout = match timeout {
            MacOsFutexTimeout::None => None,
            MacOsFutexTimeout::Relative { clock_op, timeout_op } => {
                let clock = this.read_scalar(clock_op)?.to_u32()?;
                let timeout = this.read_scalar(timeout_op)?.to_u64()?;
                Some((clock, TimeoutAnchor::Relative, timeout))
            }
            MacOsFutexTimeout::Absolute { clock_op, timeout_op } => {
                let clock = this.read_scalar(clock_op)?.to_u32()?;
                let timeout = this.read_scalar(timeout_op)?.to_u64()?;
                Some((clock, TimeoutAnchor::Absolute, timeout))
            }
        };

        // Perform validation of the arguments.
        let addr = ptr.addr().bytes();
        if addr == 0
            || !matches!(size, 4 | 8)
            || !addr.is_multiple_of(size)
            || (flags != none && flags != shared)
            || clock_timeout
                .is_some_and(|(clock, _, timeout)| clock != absolute_clock || timeout == 0)
        {
            this.set_last_error_and_return(LibcError("EINVAL"), dest)?;
            return interp_ok(());
        }

        let is_shared = flags == shared;
        let timeout = clock_timeout.map(|(_, anchor, timeout)| {
            // The only clock that is currenlty supported is the monotonic clock.
            // While the deadline argument of `os_sync_wait_on_address_with_deadline`
            // is actually not in nanoseconds but in the units of `mach_current_time`,
            // the two are equivalent in miri.
            (TimeoutClock::Monotonic, anchor, Duration::from_nanos(timeout))
        });

        // See the Linux futex implementation for why this fence exists.
        this.atomic_fence(AtomicFenceOrd::SeqCst)?;

        let layout = this.machine.layouts.uint(Size::from_bytes(size)).unwrap();
        let futex_val = this
            .read_scalar_atomic(&this.ptr_to_mplace(ptr, layout), AtomicReadOrd::Acquire)?
            .to_bits(Size::from_bytes(size))?;

        let futex = this
            .get_sync_or_init(ptr, |_| {
                MacOsFutex {
                    futex: Default::default(),
                    size: Cell::new(size),
                    shared: Cell::new(is_shared),
                }
            })
            .unwrap();

        // Detect mismatches between the flags and sizes used on this address
        // by comparing it with the parameters used by the other waiters in
        // the current list. If the list is currently empty, update those
        // parameters.
        if futex.futex.waiters() == 0 {
            futex.size.set(size);
            futex.shared.set(is_shared);
        } else if futex.size.get() != size || futex.shared.get() != is_shared {
            this.set_last_error_and_return(LibcError("EINVAL"), dest)?;
            return interp_ok(());
        }

        if futex_val == value.into() {
            // If the values are the same, we have to block.
            let futex_ref = futex.futex.clone();
            let dest = dest.clone();
            this.futex_wait(
                futex_ref.clone(),
                u32::MAX, // bitset
                timeout,
                callback!(
                    @capture<'tcx> {
                        dest: MPlaceTy<'tcx>,
                        futex_ref: FutexRef,
                    }
                    |this, unblock: UnblockKind| {
                        match unblock {
                            UnblockKind::Ready => {
                                let remaining = futex_ref.waiters().try_into().unwrap();
                                this.write_scalar(Scalar::from_i32(remaining), &dest)
                            }
                            UnblockKind::TimedOut => {
                                this.set_last_error_and_return(LibcError("ETIMEDOUT"), &dest)
                            }
                        }
                    }
                ),
            );
        } else {
            // else retrieve the current number of waiters.
            let waiters = futex.futex.waiters().try_into().unwrap();
            this.write_scalar(Scalar::from_i32(waiters), dest)?;
        }

        interp_ok(())
    }

    /// Implements [`os_sync_wake_by_address_all`] and [`os_sync_wake_by_address_any`].
    ///
    /// [`os_sync_wake_by_address_all`]: https://developer.apple.com/documentation/os/os_sync_wake_by_address_all?language=objc
    /// [`os_sync_wake_by_address_any`]: https://developer.apple.com/documentation/os/os_sync_wake_by_address_any?language=objc
    fn os_sync_wake_by_address(
        &mut self,
        addr_op: &OpTy<'tcx>,
        size_op: &OpTy<'tcx>,
        flags_op: &OpTy<'tcx>,
        all: bool,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();
        let none = this.eval_libc_u32("OS_SYNC_WAKE_BY_ADDRESS_NONE");
        let shared = this.eval_libc_u32("OS_SYNC_WAKE_BY_ADDRESS_SHARED");

        let ptr = this.read_pointer(addr_op)?;
        let size = this.read_target_usize(size_op)?;
        let flags = this.read_scalar(flags_op)?.to_u32()?;

        // Perform validation of the arguments.
        let addr = ptr.addr().bytes();
        if addr == 0 || !matches!(size, 4 | 8) || (flags != none && flags != shared) {
            this.set_last_error_and_return(LibcError("EINVAL"), dest)?;
            return interp_ok(());
        }

        let is_shared = flags == shared;

        let Some(futex) = this.get_sync_or_init(ptr, |_| {
            MacOsFutex {
                futex: Default::default(),
                size: Cell::new(size),
                shared: Cell::new(is_shared),
            }
        }) else {
            // No AllocId, or no live allocation at that AllocId. Return an
            // error code. (That seems nicer than silently doing something
            // non-intuitive.) This means that if an address gets reused by a
            // new allocation, we'll use an independent futex queue for this...
            // that seems acceptable.
            this.set_last_error_and_return(LibcError("ENOENT"), dest)?;
            return interp_ok(());
        };

        if futex.futex.waiters() == 0 {
            this.set_last_error_and_return(LibcError("ENOENT"), dest)?;
            return interp_ok(());
        // If there are waiters in the queue, they have all used the parameters
        // stored in `futex` (we check this in `os_sync_wait_on_address` above).
        // Detect mismatches between "our" parameters and the parameters used by
        // the waiters and return an error in that case.
        } else if futex.size.get() != size || futex.shared.get() != is_shared {
            this.set_last_error_and_return(LibcError("EINVAL"), dest)?;
            return interp_ok(());
        }

        let futex_ref = futex.futex.clone();

        // See the Linux futex implementation for why this fence exists.
        this.atomic_fence(AtomicFenceOrd::SeqCst)?;
        this.futex_wake(&futex_ref, u32::MAX, if all { usize::MAX } else { 1 })?;
        this.write_scalar(Scalar::from_i32(0), dest)?;
        interp_ok(())
    }

    fn os_unfair_lock_lock(&mut self, lock_op: &OpTy<'tcx>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let MacOsUnfairLock::Active { mutex_ref } = this.os_unfair_lock_get_data(lock_op)? else {
            // Trying to get a poisoned lock. Just block forever...
            this.block_thread(
                BlockReason::Sleep,
                None,
                callback!(
                    @capture<'tcx> {}
                    |_this, _unblock: UnblockKind| {
                        panic!("we shouldn't wake up ever")
                    }
                ),
            );
            return interp_ok(());
        };
        let mutex_ref = mutex_ref.clone();

        if let Some(owner) = mutex_ref.owner() {
            if owner == this.active_thread() {
                // Matching the current macOS implementation: abort on reentrant locking.
                throw_machine_stop!(TerminationInfo::Abort(
                    "attempted to lock an os_unfair_lock that is already locked by the current thread".to_owned()
                ));
            }

            this.mutex_enqueue_and_block(mutex_ref, None);
        } else {
            this.mutex_lock(&mutex_ref);
        }

        interp_ok(())
    }

    fn os_unfair_lock_trylock(
        &mut self,
        lock_op: &OpTy<'tcx>,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let MacOsUnfairLock::Active { mutex_ref } = this.os_unfair_lock_get_data(lock_op)? else {
            // Trying to get a poisoned lock. That never works.
            this.write_scalar(Scalar::from_bool(false), dest)?;
            return interp_ok(());
        };
        let mutex_ref = mutex_ref.clone();

        if mutex_ref.owner().is_some() {
            // Contrary to the blocking lock function, this does not check for
            // reentrancy.
            this.write_scalar(Scalar::from_bool(false), dest)?;
        } else {
            this.mutex_lock(&mutex_ref);
            this.write_scalar(Scalar::from_bool(true), dest)?;
        }

        interp_ok(())
    }

    fn os_unfair_lock_unlock(&mut self, lock_op: &OpTy<'tcx>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let MacOsUnfairLock::Active { mutex_ref } = this.os_unfair_lock_get_data(lock_op)? else {
            // The lock is poisoned, who knows who owns it... we'll pretend: someone else.
            throw_machine_stop!(TerminationInfo::Abort(
                "attempted to unlock an os_unfair_lock not owned by the current thread".to_owned()
            ));
        };
        let mutex_ref = mutex_ref.clone();

        // Now, unlock.
        if this.mutex_unlock(&mutex_ref)?.is_none() {
            // Matching the current macOS implementation: abort.
            throw_machine_stop!(TerminationInfo::Abort(
                "attempted to unlock an os_unfair_lock not owned by the current thread".to_owned()
            ));
        }

        // If the lock is not locked by anyone now, it went quiet.
        // Reset to zero so that it can be moved and initialized again for the next phase.
        if mutex_ref.owner().is_none() {
            let lock_place = this.deref_pointer_as(lock_op, this.machine.layouts.u32)?;
            this.write_scalar_atomic(Scalar::from_u32(0), &lock_place, AtomicWriteOrd::Relaxed)?;
        }

        interp_ok(())
    }

    fn os_unfair_lock_assert_owner(&mut self, lock_op: &OpTy<'tcx>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let MacOsUnfairLock::Active { mutex_ref } = this.os_unfair_lock_get_data(lock_op)? else {
            // The lock is poisoned, who knows who owns it... we'll pretend: someone else.
            throw_machine_stop!(TerminationInfo::Abort(
                "called os_unfair_lock_assert_owner on an os_unfair_lock not owned by the current thread".to_owned()
            ));
        };
        let mutex_ref = mutex_ref.clone();

        if mutex_ref.owner().is_none_or(|o| o != this.active_thread()) {
            throw_machine_stop!(TerminationInfo::Abort(
                "called os_unfair_lock_assert_owner on an os_unfair_lock not owned by the current thread".to_owned()
            ));
        }

        // The lock is definitely not quiet since we are the owner.

        interp_ok(())
    }

    fn os_unfair_lock_assert_not_owner(&mut self, lock_op: &OpTy<'tcx>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        let MacOsUnfairLock::Active { mutex_ref } = this.os_unfair_lock_get_data(lock_op)? else {
            // The lock is poisoned, who knows who owns it... we'll pretend: someone else.
            return interp_ok(());
        };
        let mutex_ref = mutex_ref.clone();

        if mutex_ref.owner().is_some_and(|o| o == this.active_thread()) {
            throw_machine_stop!(TerminationInfo::Abort(
                "called os_unfair_lock_assert_not_owner on an os_unfair_lock owned by the current thread".to_owned()
            ));
        }

        // If the lock is not locked by anyone now, it went quiet.
        // Reset to zero so that it can be moved and initialized again for the next phase.
        if mutex_ref.owner().is_none() {
            let lock_place = this.deref_pointer_as(lock_op, this.machine.layouts.u32)?;
            this.write_scalar_atomic(Scalar::from_u32(0), &lock_place, AtomicWriteOrd::Relaxed)?;
        }

        interp_ok(())
    }
}
