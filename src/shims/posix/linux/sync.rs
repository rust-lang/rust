use crate::*;
use rustc_target::abi::{Align, Size};

/// Implementation of the SYS_futex syscall.
pub fn futex<'tcx>(
    this: &mut MiriEvalContext<'_, 'tcx>,
    args: &[OpTy<'tcx, Tag>],
    dest: PlaceTy<'tcx, Tag>,
) -> InterpResult<'tcx> {
    // The amount of arguments used depends on the type of futex operation.
    // The full futex syscall takes six arguments (excluding the syscall
    // number), which is also the maximum amount of arguments a linux syscall
    // can take on most architectures.
    // However, not all futex operations use all six arguments. The unused ones
    // may or may not be left out from the `syscall()` call.
    // Therefore we don't use `check_arg_count` here, but only check for the
    // number of arguments to fall within a range.
    if !(4..=7).contains(&args.len()) {
        throw_ub_format!("incorrect number of arguments for futex syscall: got {}, expected between 4 and 7 (inclusive)", args.len());
    }

    // The first three arguments (after the syscall number itself) are the same to all futex operations:
    //     (int *addr, int op, int val).
    // We checked above that these definitely exist.
    //
    // `addr` is used to identify the mutex, but note that not all futex
    // operations actually read from this addres or even require this address
    // to exist. Also, the type of `addr` is not consistent. The API requires
    // it to be a 4-byte aligned pointer, and will use the 4 bytes at the given
    // address as an (atomic) i32. It's not uncommon for `addr` to be passed as
    // another type than `*mut i32`, such as `*const AtomicI32`.
    let addr = this.force_ptr(this.read_scalar(args[1])?.check_init()?)?;
    let op = this.read_scalar(args[2])?.to_i32()?;
    let val = this.read_scalar(args[3])?.to_i32()?;

    let thread = this.get_active_thread();

    let futex_private = this.eval_libc_i32("FUTEX_PRIVATE_FLAG")?;
    let futex_wait = this.eval_libc_i32("FUTEX_WAIT")?;
    let futex_wake = this.eval_libc_i32("FUTEX_WAKE")?;

    // FUTEX_PRIVATE enables an optimization that stops it from working across processes.
    // Miri doesn't support that anyway, so we ignore that flag.
    match op & !futex_private {
        // FUTEX_WAIT: (int *addr, int op = FUTEX_WAIT, int val, const timespec *timeout)
        // Blocks the thread if *addr still equals val. Wakes up when FUTEX_WAKE is called on the same address,
        // or *timeout expires. `timeout == null` for an infinite timeout.
        op if op == futex_wait => {
            if args.len() < 5 {
                throw_ub_format!("incorrect number of arguments for FUTEX_WAIT syscall: got {}, expected at least 5", args.len());
            }
            let timeout = this.read_scalar(args[4])?.check_init()?;
            if !this.is_null(timeout)? {
                // FIXME: Implement timeouts. The condvar waiting code is probably a good example to start with.
                // Note that a triggered timeout should have this syscall return with -1 and errno set to ETIMEOUT.
                throw_ub_format!("miri does not support timeouts for futex operations");
            }
            // Check the pointer for alignment and validity.
            // Atomic operations are only available for fully aligned values.
            this.memory.check_ptr_access(addr.into(), Size::from_bytes(4), Align::from_bytes(4).unwrap())?;
            // Read an `i32` through the pointer, regardless of any wrapper types (e.g. `AtomicI32`).
            let futex_val = this.memory.get_raw(addr.alloc_id)?.read_scalar(this, addr, Size::from_bytes(4))?.to_i32()?;
            if val == futex_val {
                // The value still matches, so we block the trait make it wait for FUTEX_WAKE.
                this.block_thread(thread);
                this.futex_wait(addr, thread);
                // Succesfully waking up from FUTEX_WAIT always returns zero.
                this.write_scalar(Scalar::from_i32(0), dest)?;
            } else {
                // The futex value doesn't match the expected value, so we return failure
                // right away without sleeping: -1 and errno set to EAGAIN.
                let eagain = this.eval_libc("EAGAIN")?;
                this.set_last_error(eagain)?;
                this.write_scalar(Scalar::from_i32(-1), dest)?;
            }
        }
        // FUTEX_WAKE: (int *addr, int op = FUTEX_WAKE, int val)
        // Wakes at most `val` threads waiting on the futex at `addr`.
        // Returns the amount of threads woken up.
        // Does not access the futex value at *addr.
        op if op == futex_wake => {
            let mut n = 0;
            for _ in 0..val {
                if let Some(thread) = this.futex_wake(addr) {
                    this.unblock_thread(thread);
                    n += 1;
                } else {
                    break;
                }
            }
            this.write_scalar(Scalar::from_i32(n), dest)?;
        }
        op => throw_unsup_format!("miri does not support SYS_futex operation {}", op),
    }

    Ok(())
}
