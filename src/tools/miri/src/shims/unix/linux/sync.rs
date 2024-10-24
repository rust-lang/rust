use std::time::Duration;

use crate::helpers::check_min_arg_count;
use crate::shims::unix::env::EvalContextExt;
use crate::*;

/// Implementation of the SYS_futex syscall.
/// `args` is the arguments *including* the syscall number.
pub fn futex<'tcx>(
    this: &mut MiriInterpCx<'tcx>,
    args: &[OpTy<'tcx>],
    dest: &MPlaceTy<'tcx>,
) -> InterpResult<'tcx> {
    // The amount of arguments used depends on the type of futex operation.
    // The full futex syscall takes six arguments (excluding the syscall
    // number), which is also the maximum amount of arguments a linux syscall
    // can take on most architectures.
    // However, not all futex operations use all six arguments. The unused ones
    // may or may not be left out from the `syscall()` call.
    // Therefore we don't use `check_arg_count` here, but only check for the
    // number of arguments to fall within a range.
    let [_, addr, op] = check_min_arg_count("`syscall(SYS_futex, ...)`", args)?;

    // The first two arguments (after the syscall number itself) are the same to all futex operations:
    //     (int *addr, int op).
    // We checked above that these definitely exist.
    let addr = this.read_pointer(addr)?;
    let op = this.read_scalar(op)?.to_i32()?;

    // This is a vararg function so we have to bring our own type for this pointer.
    let addr = this.ptr_to_mplace(addr, this.machine.layouts.i32);
    let addr_usize = addr.ptr().addr().bytes();

    let futex_private = this.eval_libc_i32("FUTEX_PRIVATE_FLAG");
    let futex_wait = this.eval_libc_i32("FUTEX_WAIT");
    let futex_wait_bitset = this.eval_libc_i32("FUTEX_WAIT_BITSET");
    let futex_wake = this.eval_libc_i32("FUTEX_WAKE");
    let futex_wake_bitset = this.eval_libc_i32("FUTEX_WAKE_BITSET");
    let futex_realtime = this.eval_libc_i32("FUTEX_CLOCK_REALTIME");
    let futex_lock_pi = this.eval_libc_i32("FUTEX_LOCK_PI");
    let futex_unlock_pi = this.eval_libc_i32("FUTEX_UNLOCK_PI");
    let futex_waiters = this.eval_libc_u32("FUTEX_WAITERS");
    let futex_tid_mask = this.eval_libc_u32("FUTEX_TID_MASK");

    // Ok(None) for EINVAL set, Ok(Some(None)) for no timeout (infinity), Ok(Some(Some(...))) for a timeout.
    // Forgive me, I don't want to create an enum for this return value.
    fn read_timeout<'tcx>(
        this: &mut MiriInterpCx<'tcx>,
        timeout: &OpTy<'tcx>,
        use_realtime_clock: bool,
        use_absolute_time: bool,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, Option<Option<(TimeoutClock, TimeoutAnchor, Duration)>>> {
        let timeout = this.deref_pointer_as(timeout, this.libc_ty_layout("timespec"))?;
        interp_ok(Some(if this.ptr_is_null(timeout.ptr())? {
            None
        } else {
            let duration = match this.read_timespec(&timeout)? {
                Some(duration) => duration,
                None => {
                    this.set_last_error(LibcError("EINVAL"))?;
                    this.write_scalar(Scalar::from_target_isize(-1, this), dest)?;
                    return interp_ok(None);
                }
            };
            let timeout_clock = if use_realtime_clock {
                this.check_no_isolation(
                    "`futex` syscall with `op=FUTEX_WAIT` and non-null timeout with `FUTEX_CLOCK_REALTIME`",
                )?;
                TimeoutClock::RealTime
            } else {
                TimeoutClock::Monotonic
            };
            let timeout_anchor = if use_absolute_time {
                // FUTEX_WAIT_BITSET uses an absolute timestamp.
                TimeoutAnchor::Absolute
            } else {
                // FUTEX_WAIT uses a relative timestamp.
                TimeoutAnchor::Relative
            };
            Some((timeout_clock, timeout_anchor, duration))
        }))
    }

    // FUTEX_PRIVATE enables an optimization that stops it from working across processes.
    // Miri doesn't support that anyway, so we ignore that flag.
    match op & !futex_private {
        // FUTEX_WAIT: (int *addr, int op = FUTEX_WAIT, int val, const timespec *timeout)
        // Blocks the thread if *addr still equals val. Wakes up when FUTEX_WAKE is called on the same address,
        // or *timeout expires. `timeout == null` for an infinite timeout.
        //
        // FUTEX_WAIT_BITSET: (int *addr, int op = FUTEX_WAIT_BITSET, int val, const timespec *timeout, int *_ignored, unsigned int bitset)
        // This is identical to FUTEX_WAIT, except:
        //  - The timeout is absolute rather than relative.
        //  - You can specify the bitset to selecting what WAKE operations to respond to.
        op if op & !futex_realtime == futex_wait || op & !futex_realtime == futex_wait_bitset => {
            let wait_bitset = op & !futex_realtime == futex_wait_bitset;

            let (val, timeout, bitset) = if wait_bitset {
                let [_, _, _, val, timeout, uaddr2, bitset] =
                    check_min_arg_count("`syscall(SYS_futex, FUTEX_WAIT_BITSET, ...)`", args)?;
                let val = this.read_scalar(val)?.to_i32()?;
                let _timeout = this.read_pointer(timeout)?;
                let _uaddr2 = this.read_pointer(uaddr2)?;
                (val, timeout, this.read_scalar(bitset)?.to_u32()?)
            } else {
                let [_, _, _, val, timeout] =
                    check_min_arg_count("`syscall(SYS_futex, FUTEX_WAIT, ...)`", args)?;
                let val = this.read_scalar(val)?.to_i32()?;
                (val, timeout, u32::MAX)
            };

            let Some(timeout) = read_timeout(
                this,
                timeout,
                op & futex_realtime == futex_realtime,
                wait_bitset,
                dest,
            )?
            else {
                return interp_ok(());
            };

            if bitset == 0 {
                this.set_last_error(LibcError("EINVAL"))?;
                this.write_scalar(Scalar::from_target_isize(-1, this), dest)?;
                return interp_ok(());
            }

            // There may be a concurrent thread changing the value of addr
            // and then invoking the FUTEX_WAKE syscall. It is critical that the
            // effects of this and the other thread are correctly observed,
            // otherwise we will deadlock.
            //
            // There are two scenarios to consider:
            // 1. If we (FUTEX_WAIT) execute first, we'll push ourselves into
            //    the waiters queue and go to sleep. They (addr write & FUTEX_WAKE)
            //    will see us in the queue and wake us up.
            // 2. If they (addr write & FUTEX_WAKE) execute first, we must observe
            //    addr's new value. If we see an outdated value that happens to equal
            //    the expected val, then we'll put ourselves to sleep with no one to wake us
            //    up, so we end up with a deadlock. This is prevented by having a SeqCst
            //    fence inside FUTEX_WAKE syscall, and another SeqCst fence
            //    below, the atomic read on addr after the SeqCst fence is guaranteed
            //    not to see any value older than the addr write immediately before
            //    calling FUTEX_WAKE. We'll see futex_val != val and return without
            //    sleeping.
            //
            //    Note that the fences do not create any happens-before relationship.
            //    The read sees the write immediately before the fence not because
            //    one happens after the other, but is instead due to a guarantee unique
            //    to SeqCst fences that restricts what an atomic read placed AFTER the
            //    fence can see. The read still has to be atomic, otherwise it's a data
            //    race. This guarantee cannot be achieved with acquire-release fences
            //    since they only talk about reads placed BEFORE a fence - and places
            //    no restrictions on what the read itself can see, only that there is
            //    a happens-before between the fences IF the read happens to see the
            //    right value. This is useless to us, since we need the read itself
            //    to see an up-to-date value.
            //
            // The above case distinction is valid since both FUTEX_WAIT and FUTEX_WAKE
            // contain a SeqCst fence, therefore inducing a total order between the operations.
            // It is also critical that the fence, the atomic load, and the comparison in FUTEX_WAIT
            // altogether happen atomically. If the other thread's fence in FUTEX_WAKE
            // gets interleaved after our fence, then we lose the guarantee on the
            // atomic load being up-to-date; if the other thread's write on addr and FUTEX_WAKE
            // call are interleaved after the load but before the comparison, then we get a TOCTOU
            // race condition, and go to sleep thinking the other thread will wake us up,
            // even though they have already finished.
            //
            // Thankfully, preemptions cannot happen inside a Miri shim, so we do not need to
            // do anything special to guarantee fence-load-comparison atomicity.
            this.atomic_fence(AtomicFenceOrd::SeqCst)?;
            // Read an `i32` through the pointer, regardless of any wrapper types.
            // It's not uncommon for `addr` to be passed as another type than `*mut i32`, such as `*const AtomicI32`.
            let futex_val = this.read_scalar_atomic(&addr, AtomicReadOrd::Relaxed)?.to_i32()?;
            if val == futex_val {
                // Check that the top waiter (if exists) is waiting using FUTEX_WAIT_*.
                if this.futex_waiter_count(addr_usize) != 0 && this.futex_top_waiter_extra(addr_usize).is_some() {
                    this.set_last_error(LibcError("EINVAL"))?;
                    this.write_scalar(Scalar::from_target_isize(-1, this), dest)?;
                    return interp_ok(());
                }

                // The value still matches, so we block the thread and make it wait for FUTEX_WAKE.
                this.futex_wait(
                    addr_usize,
                    bitset,
                    timeout,
                    Scalar::from_target_isize(0, this), // retval_succ
                    Scalar::from_target_isize(-1, this), // retval_timeout
                    None,
                    dest.clone(),
                    this.eval_libc("ETIMEDOUT"),
                );
            } else {
                // The futex value doesn't match the expected value, so we return failure
                // right away without sleeping: -1 and errno set to EAGAIN.
                let eagain = this.eval_libc("EAGAIN");
                this.set_last_error(eagain)?;
                this.write_scalar(Scalar::from_target_isize(-1, this), dest)?;
            }
        }
        // FUTEX_WAKE: (int *addr, int op = FUTEX_WAKE, int val)
        // Wakes at most `val` threads waiting on the futex at `addr`.
        // Returns the amount of threads woken up.
        // Does not access the futex value at *addr.
        // FUTEX_WAKE_BITSET: (int *addr, int op = FUTEX_WAKE, int val, const timespect *_unused, int *_unused, unsigned int bitset)
        // Same as FUTEX_WAKE, but allows you to specify a bitset to select which threads to wake up.
        op if op == futex_wake || op == futex_wake_bitset => {
            let (val, bitset) = if op == futex_wake_bitset {
                let [_, _, _, val, timeout, uaddr2, bitset] =
                    check_min_arg_count("`syscall(SYS_futex, FUTEX_WAKE_BITSET, ...)`", args)?;
                let val = this.read_scalar(val)?.to_u32()?;
                let _timeout = this.read_pointer(timeout)?;
                let _uaddr2 = this.read_pointer(uaddr2)?;
                (val, this.read_scalar(bitset)?.to_u32()?)
            } else {
                let [_, _, _, val] =
                    check_min_arg_count("`syscall(SYS_futex, FUTEX_WAKE, ...)`", args)?;
                let val = this.read_scalar(val)?.to_u32()?;
                (val, u32::MAX)
            };
            if bitset == 0 {
                this.set_last_error(LibcError("EINVAL"))?;
                this.write_scalar(Scalar::from_target_isize(-1, this), dest)?;
                return interp_ok(());
            }
            // Together with the SeqCst fence in futex_wait, this makes sure that futex_wait
            // will see the latest value on addr which could be changed by our caller
            // before doing the syscall.
            this.atomic_fence(AtomicFenceOrd::SeqCst)?;
            let mut n = 0;

            // Check that the waiter is waiting using FUTEX_WAIT_*.
            if this.futex_waiter_count(addr_usize) != 0 {
                if this.futex_top_waiter_extra(addr_usize).is_some() {
                    this.set_last_error(LibcError("EINVAL"))?;
                    this.write_scalar(Scalar::from_target_isize(-1, this), dest)?;
                    return interp_ok(());
                }
            }

            #[allow(clippy::arithmetic_side_effects)]
            for _ in 0..val {
                if this.futex_wake(addr_usize, bitset)? {
                    n += 1;
                } else {
                    break;
                }
            }
            this.write_scalar(Scalar::from_target_isize(n, this), dest)?;
        }
        op if op == futex_lock_pi => {
            let [_, _, _, _, timeout] =
                check_min_arg_count("`syscall(SYS_futex, FUTEX_LOCK_PI, ...)`", args)?;

            // FUTEX_LOCK_PI uses absolute CLOCK_REALTIME timestamp.
            let Some(timeout) = read_timeout(this, timeout, true, true, dest)? else {
                return interp_ok(());
            };

            // The tid of the owner is store to *addr.
            // N.B. it is not the same as posix thread id.
            let tid = this.linux_gettid()?.to_u32()?;

            // The same as above. This makes modifications visible to us.
            this.atomic_fence(AtomicFenceOrd::SeqCst)?;

            // For bitand working properly, we read it as a u32.
            let futex_val = this.read_scalar_atomic(&addr, AtomicReadOrd::Relaxed)?.to_u32()?;

            if futex_val == 0 {
                // 0 means unlocked - then lock it.

                // Check that there is no waiters.
                if this.futex_waiter_count(addr_usize) != 0 {
                    this.set_last_error(LibcError("EINVAL"))?;
                    this.write_scalar(Scalar::from_target_isize(-1, this), dest)?;
                    return interp_ok(());
                }

                // Declare ownership.
                this.write_scalar_atomic(Scalar::from_u32(tid), &addr, AtomicWriteOrd::Relaxed)?;

                // This ensures all loads afterwards get updated value of *addr.
                this.atomic_fence(AtomicFenceOrd::SeqCst)?;

                // FUTEX_LOCK_PI returns 0 on success.
                this.write_scalar(Scalar::from_target_isize(0, this), dest)?;
            } else if (futex_val & futex_tid_mask) == tid {
                // Locked by self.
                this.set_last_error(LibcError("EDEADLK"))?;
                this.write_scalar(Scalar::from_target_isize(-1, this), dest)?;
            } else {
                // Other values mean locked.

                if this.futex_waiter_count(addr_usize) == 0 {
                    // Mark the futex as contended.
                    this.write_scalar_atomic(
                        Scalar::from_u32(futex_val | futex_waiters),
                        &addr,
                        AtomicWriteOrd::Relaxed,
                    )?;

                    // This ensures all loads afterwards get updated value of *addr.
                    this.atomic_fence(AtomicFenceOrd::SeqCst)?;
                } else {
                    // Check that the futex has been marked as contended.
                    // Check that the top waiter is waiting using FUTEX_LOCK_PI.
                    if (futex_val & futex_waiters) == 0 || this.futex_top_waiter_extra(addr_usize).is_none() {
                        this.set_last_error(LibcError("EINVAL"))?;
                        this.write_scalar(Scalar::from_target_isize(-1, this), dest)?;
                        return interp_ok(());
                    }
                }

                // Put ourselves into the wait queue.
                this.futex_wait(
                    addr_usize,
                    u32::MAX,
                    timeout,
                    Scalar::from_target_isize(0, this), // retval_succ
                    Scalar::from_target_isize(-1, this), // retval_timeout
                    Some(tid),
                    dest.clone(),
                    this.eval_libc("ETIMEDOUT"),
                );
            }
        }
        op if op == futex_unlock_pi => {
            let tid = this.linux_gettid()?.to_u32()?;

            // This ensures all modifications happen before.
            this.atomic_fence(AtomicFenceOrd::SeqCst)?;

            // Check we owns lock.
            let futex_val = this.read_scalar_atomic(&addr, AtomicReadOrd::Relaxed)?.to_u32()?;
            if (futex_val & futex_tid_mask) != tid {
                this.set_last_error(LibcError("EINVAL"))?;
                this.write_scalar(Scalar::from_target_isize(-1, this), dest)?;
                return interp_ok(());
            }

            let waiter_count = this.futex_waiter_count(addr_usize);
            let new_futex_val = if waiter_count == 0 {
                0
            } else {
                let Some(new_tid) = this.futex_top_waiter_extra(addr_usize) else {
                    // The waiter is not waiting using FUTEX_LOCK_PI.
                    this.set_last_error(LibcError("EINVAL"))?;
                    this.write_scalar(Scalar::from_target_isize(-1, this), dest)?;
                    return interp_ok(());
                };

                if waiter_count > 1 {
                    // Still contended after we unlocks.
                    new_tid | futex_waiters
                } else {
                    new_tid
                }
            };

            // Update locked state.
            this.write_scalar_atomic(
                Scalar::from_u32(new_futex_val),
                &addr,
                AtomicWriteOrd::Relaxed,
            )?;

            // This ensures all loads afterwards get updated value of *addr.
            // There are no preemptions so no one can wake after we set the futex to unlocked
            // and before we use futex_wake to wake one waiter.
            this.atomic_fence(AtomicFenceOrd::SeqCst)?;

            // Unlocking wakes zero or one waiters.
            let _ = this.futex_wake(addr_usize, u32::MAX)?;

            // FUTEX_UNLOCK_PI returns 0 on success.
            this.write_scalar(Scalar::from_target_isize(0, this), dest)?;
        }
        op => throw_unsup_format!("Miri does not support `futex` syscall with op={}", op),
    }

    interp_ok(())
}
