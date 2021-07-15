use crate::thread::Time;
use crate::*;
use rustc_target::abi::{Align, Size};
use std::time::{Instant, SystemTime};

/// Implementation of the SYS_futex syscall.
pub fn futex<'tcx>(
    this: &mut MiriEvalContext<'_, 'tcx>,
    args: &[OpTy<'tcx, Tag>],
    dest: &PlaceTy<'tcx, Tag>,
) -> InterpResult<'tcx> {
    // The amount of arguments used depends on the type of futex operation.
    // The full futex syscall takes six arguments (excluding the syscall
    // number), which is also the maximum amount of arguments a linux syscall
    // can take on most architectures.
    // However, not all futex operations use all six arguments. The unused ones
    // may or may not be left out from the `syscall()` call.
    // Therefore we don't use `check_arg_count` here, but only check for the
    // number of arguments to fall within a range.
    if args.len() < 4 {
        throw_ub_format!(
            "incorrect number of arguments for `futex` syscall: got {}, expected at least 4",
            args.len()
        );
    }

    // The first three arguments (after the syscall number itself) are the same to all futex operations:
    //     (int *addr, int op, int val).
    // We checked above that these definitely exist.
    let addr = this.read_immediate(&args[1])?;
    let op = this.read_scalar(&args[2])?.to_i32()?;
    let val = this.read_scalar(&args[3])?.to_i32()?;

    let thread = this.get_active_thread();
    let addr_scalar = addr.to_scalar()?;

    let futex_private = this.eval_libc_i32("FUTEX_PRIVATE_FLAG")?;
    let futex_wait = this.eval_libc_i32("FUTEX_WAIT")?;
    let futex_wake = this.eval_libc_i32("FUTEX_WAKE")?;
    let futex_realtime = this.eval_libc_i32("FUTEX_CLOCK_REALTIME")?;

    // FUTEX_PRIVATE enables an optimization that stops it from working across processes.
    // Miri doesn't support that anyway, so we ignore that flag.
    match op & !futex_private {
        // FUTEX_WAIT: (int *addr, int op = FUTEX_WAIT, int val, const timespec *timeout)
        // Blocks the thread if *addr still equals val. Wakes up when FUTEX_WAKE is called on the same address,
        // or *timeout expires. `timeout == null` for an infinite timeout.
        op if op & !futex_realtime == futex_wait => {
            if args.len() < 5 {
                throw_ub_format!(
                    "incorrect number of arguments for `futex` syscall with `op=FUTEX_WAIT`: got {}, expected at least 5",
                    args.len()
                );
            }

            // `deref_operand` but not actually dereferencing the ptr yet (it might be NULL!).
            let timeout = this.ref_to_mplace(&this.read_immediate(&args[4])?)?;
            let timeout_time = if this.ptr_is_null(timeout.ptr)? {
                None
            } else {
                this.check_no_isolation(
                    "`futex` syscall with `op=FUTEX_WAIT` and non-null timeout",
                )?;
                let duration = match this.read_timespec(&timeout)? {
                    Some(duration) => duration,
                    None => {
                        let einval = this.eval_libc("EINVAL")?;
                        this.set_last_error(einval)?;
                        this.write_scalar(Scalar::from_machine_isize(-1, this), dest)?;
                        return Ok(());
                    }
                };
                Some(if op & futex_realtime != 0 {
                    Time::RealTime(SystemTime::now().checked_add(duration).unwrap())
                } else {
                    Time::Monotonic(Instant::now().checked_add(duration).unwrap())
                })
            };
            // Check the pointer for alignment and validity.
            // The API requires `addr` to be a 4-byte aligned pointer, and will
            // use the 4 bytes at the given address as an (atomic) i32.
            this.memory.check_ptr_access_align(
                this.scalar_to_ptr(addr_scalar),
                Size::from_bytes(4),
                Align::from_bytes(4).unwrap(),
                CheckInAllocMsg::MemoryAccessTest,
            )?;
            // Read an `i32` through the pointer, regardless of any wrapper types.
            // It's not uncommon for `addr` to be passed as another type than `*mut i32`, such as `*const AtomicI32`.
            // FIXME: this fails if `addr` is not a pointer type.
            // The atomic ordering for futex(https://man7.org/linux/man-pages/man2/futex.2.html):
            //  "The load of the value of the futex word is an
            //   atomic memory access (i.e., using atomic machine instructions
            //   of the respective architecture).  This load, the comparison
            //   with the expected value, and starting to sleep are performed
            //   atomically and totally ordered with respect to other futex
            //   operations on the same futex word."
            // SeqCst is total order over all operations.
            // FIXME: check if this should be changed when weak memory orders are added.
            let futex_val = this
                .read_scalar_at_offset_atomic(
                    &addr.into(),
                    0,
                    this.machine.layouts.i32,
                    AtomicReadOp::SeqCst,
                )?
                .to_i32()?;
            if val == futex_val {
                // The value still matches, so we block the trait make it wait for FUTEX_WAKE.
                this.block_thread(thread);
                this.futex_wait(addr_scalar.to_machine_usize(this)?, thread);
                // Succesfully waking up from FUTEX_WAIT always returns zero.
                this.write_scalar(Scalar::from_machine_isize(0, this), dest)?;
                // Register a timeout callback if a timeout was specified.
                // This callback will override the return value when the timeout triggers.
                let dest = *dest;
                if let Some(timeout_time) = timeout_time {
                    this.register_timeout_callback(
                        thread,
                        timeout_time,
                        Box::new(move |this| {
                            this.unblock_thread(thread);
                            this.futex_remove_waiter(addr_scalar.to_machine_usize(this)?, thread);
                            let etimedout = this.eval_libc("ETIMEDOUT")?;
                            this.set_last_error(etimedout)?;
                            this.write_scalar(Scalar::from_machine_isize(-1, this), &dest)?;
                            Ok(())
                        }),
                    );
                }
            } else {
                // The futex value doesn't match the expected value, so we return failure
                // right away without sleeping: -1 and errno set to EAGAIN.
                let eagain = this.eval_libc("EAGAIN")?;
                this.set_last_error(eagain)?;
                this.write_scalar(Scalar::from_machine_isize(-1, this), dest)?;
            }
        }
        // FUTEX_WAKE: (int *addr, int op = FUTEX_WAKE, int val)
        // Wakes at most `val` threads waiting on the futex at `addr`.
        // Returns the amount of threads woken up.
        // Does not access the futex value at *addr.
        op if op == futex_wake => {
            let mut n = 0;
            for _ in 0..val {
                if let Some(thread) = this.futex_wake(addr_scalar.to_machine_usize(this)?) {
                    this.unblock_thread(thread);
                    this.unregister_timeout_callback_if_exists(thread);
                    n += 1;
                } else {
                    break;
                }
            }
            this.write_scalar(Scalar::from_machine_isize(n, this), dest)?;
        }
        op => throw_unsup_format!("Miri does not support `futex` syscall with op={}", op),
    }

    Ok(())
}
