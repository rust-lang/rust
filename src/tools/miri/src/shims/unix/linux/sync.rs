use std::time::SystemTime;

use crate::concurrency::thread::{MachineCallback, Time};
use crate::*;

/// Implementation of the SYS_futex syscall.
/// `args` is the arguments *after* the syscall number.
pub fn futex<'tcx>(
    this: &mut MiriInterpCx<'_, 'tcx>,
    args: &[OpTy<'tcx, Provenance>],
    dest: &PlaceTy<'tcx, Provenance>,
) -> InterpResult<'tcx> {
    // The amount of arguments used depends on the type of futex operation.
    // The full futex syscall takes six arguments (excluding the syscall
    // number), which is also the maximum amount of arguments a linux syscall
    // can take on most architectures.
    // However, not all futex operations use all six arguments. The unused ones
    // may or may not be left out from the `syscall()` call.
    // Therefore we don't use `check_arg_count` here, but only check for the
    // number of arguments to fall within a range.
    if args.len() < 3 {
        throw_ub_format!(
            "incorrect number of arguments for `futex` syscall: got {}, expected at least 3",
            args.len()
        );
    }

    // The first three arguments (after the syscall number itself) are the same to all futex operations:
    //     (int *addr, int op, int val).
    // We checked above that these definitely exist.
    let addr = this.read_pointer(&args[0])?;
    let op = this.read_scalar(&args[1])?.to_i32()?;
    let val = this.read_scalar(&args[2])?.to_i32()?;

    let thread = this.get_active_thread();
    // This is a vararg function so we have to bring our own type for this pointer.
    let addr = MPlaceTy::from_aligned_ptr(addr, this.machine.layouts.i32);
    let addr_usize = addr.ptr.addr().bytes();

    let futex_private = this.eval_libc_i32("FUTEX_PRIVATE_FLAG");
    let futex_wait = this.eval_libc_i32("FUTEX_WAIT");
    let futex_wait_bitset = this.eval_libc_i32("FUTEX_WAIT_BITSET");
    let futex_wake = this.eval_libc_i32("FUTEX_WAKE");
    let futex_wake_bitset = this.eval_libc_i32("FUTEX_WAKE_BITSET");
    let futex_realtime = this.eval_libc_i32("FUTEX_CLOCK_REALTIME");

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

            let bitset = if wait_bitset {
                if args.len() < 6 {
                    throw_ub_format!(
                        "incorrect number of arguments for `futex` syscall with `op=FUTEX_WAIT_BITSET`: got {}, expected at least 6",
                        args.len()
                    );
                }
                let _timeout = this.read_pointer(&args[3])?;
                let _uaddr2 = this.read_pointer(&args[4])?;
                this.read_scalar(&args[5])?.to_u32()?
            } else {
                if args.len() < 4 {
                    throw_ub_format!(
                        "incorrect number of arguments for `futex` syscall with `op=FUTEX_WAIT`: got {}, expected at least 4",
                        args.len()
                    );
                }
                u32::MAX
            };

            if bitset == 0 {
                let einval = this.eval_libc("EINVAL");
                this.set_last_error(einval)?;
                this.write_scalar(Scalar::from_target_isize(-1, this), dest)?;
                return Ok(());
            }

            // `deref_operand` but not actually dereferencing the ptr yet (it might be NULL!).
            let timeout = this.ref_to_mplace(&this.read_immediate(&args[3])?)?;
            let timeout_time = if this.ptr_is_null(timeout.ptr)? {
                None
            } else {
                let realtime = op & futex_realtime == futex_realtime;
                if realtime {
                    this.check_no_isolation(
                        "`futex` syscall with `op=FUTEX_WAIT` and non-null timeout with `FUTEX_CLOCK_REALTIME`",
                    )?;
                }
                let duration = match this.read_timespec(&timeout)? {
                    Some(duration) => duration,
                    None => {
                        let einval = this.eval_libc("EINVAL");
                        this.set_last_error(einval)?;
                        this.write_scalar(Scalar::from_target_isize(-1, this), dest)?;
                        return Ok(());
                    }
                };
                Some(if wait_bitset {
                    // FUTEX_WAIT_BITSET uses an absolute timestamp.
                    if realtime {
                        Time::RealTime(SystemTime::UNIX_EPOCH.checked_add(duration).unwrap())
                    } else {
                        Time::Monotonic(this.machine.clock.anchor().checked_add(duration).unwrap())
                    }
                } else {
                    // FUTEX_WAIT uses a relative timestamp.
                    if realtime {
                        Time::RealTime(SystemTime::now().checked_add(duration).unwrap())
                    } else {
                        Time::Monotonic(this.machine.clock.now().checked_add(duration).unwrap())
                    }
                })
            };
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
            // contain a SeqCst fence, therefore inducting a total order between the operations.
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
                // The value still matches, so we block the thread make it wait for FUTEX_WAKE.
                this.block_thread(thread);
                this.futex_wait(addr_usize, thread, bitset);
                // Succesfully waking up from FUTEX_WAIT always returns zero.
                this.write_scalar(Scalar::from_target_isize(0, this), dest)?;
                // Register a timeout callback if a timeout was specified.
                // This callback will override the return value when the timeout triggers.
                if let Some(timeout_time) = timeout_time {
                    struct Callback<'tcx> {
                        thread: ThreadId,
                        addr_usize: u64,
                        dest: PlaceTy<'tcx, Provenance>,
                    }

                    impl<'tcx> VisitTags for Callback<'tcx> {
                        fn visit_tags(&self, visit: &mut dyn FnMut(BorTag)) {
                            let Callback { thread: _, addr_usize: _, dest } = self;
                            dest.visit_tags(visit);
                        }
                    }

                    impl<'mir, 'tcx: 'mir> MachineCallback<'mir, 'tcx> for Callback<'tcx> {
                        fn call(&self, this: &mut MiriInterpCx<'mir, 'tcx>) -> InterpResult<'tcx> {
                            this.unblock_thread(self.thread);
                            this.futex_remove_waiter(self.addr_usize, self.thread);
                            let etimedout = this.eval_libc("ETIMEDOUT");
                            this.set_last_error(etimedout)?;
                            this.write_scalar(Scalar::from_target_isize(-1, this), &self.dest)?;

                            Ok(())
                        }
                    }

                    this.register_timeout_callback(
                        thread,
                        timeout_time,
                        Box::new(Callback { thread, addr_usize, dest: dest.clone() }),
                    );
                }
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
            let bitset = if op == futex_wake_bitset {
                if args.len() < 6 {
                    throw_ub_format!(
                        "incorrect number of arguments for `futex` syscall with `op=FUTEX_WAKE_BITSET`: got {}, expected at least 6",
                        args.len()
                    );
                }
                let _timeout = this.read_pointer(&args[3])?;
                let _uaddr2 = this.read_pointer(&args[4])?;
                this.read_scalar(&args[5])?.to_u32()?
            } else {
                u32::MAX
            };
            if bitset == 0 {
                let einval = this.eval_libc("EINVAL");
                this.set_last_error(einval)?;
                this.write_scalar(Scalar::from_target_isize(-1, this), dest)?;
                return Ok(());
            }
            // Together with the SeqCst fence in futex_wait, this makes sure that futex_wait
            // will see the latest value on addr which could be changed by our caller
            // before doing the syscall.
            this.atomic_fence(AtomicFenceOrd::SeqCst)?;
            let mut n = 0;
            #[allow(clippy::integer_arithmetic)]
            for _ in 0..val {
                if let Some(thread) = this.futex_wake(addr_usize, bitset) {
                    this.unblock_thread(thread);
                    this.unregister_timeout_callback_if_exists(thread);
                    n += 1;
                } else {
                    break;
                }
            }
            this.write_scalar(Scalar::from_target_isize(n, this), dest)?;
        }
        op => throw_unsup_format!("Miri does not support `futex` syscall with op={}", op),
    }

    Ok(())
}
