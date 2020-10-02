use crate::*;
use rustc_target::abi::{Align, Size};

/// Implementation of the SYS_futex syscall.
pub fn futex<'tcx>(
    this: &mut MiriEvalContext<'_, 'tcx>,
    args: &[OpTy<'tcx, Tag>],
    dest: PlaceTy<'tcx, Tag>,
) -> InterpResult<'tcx> {
    // The amount of arguments used depends on the type of futex operation.
    // Some users always pass all arguments, even the unused ones, due to how they wrap this syscall in their code base.
    // Some other users pass only the arguments the operation actually needs. So we don't use `check_arg_count` here.
    if !(4..=7).contains(&args.len()) {
        throw_ub_format!("incorrect number of arguments for futex syscall: got {}, expected between 4 and 7 (inclusive)", args.len());
    }

    // The first three arguments (after the syscall number itself) are the same to all futex operations:
    //     (int *addr, int op, int val).
    // Although note that the first one is often passed as a different pointer type, e.g. `*const AtomicU32` or `*mut u32`.
    let addr = this.deref_operand(args[1])?;
    let op = this.read_scalar(args[2])?.to_i32()?;
    let val = this.read_scalar(args[3])?.to_i32()?;

    // The raw pointer value is used to identify the mutex.
    // Not all mutex operations actually read from this address or even require this address to exist.
    let futex_ptr = addr.ptr.assert_ptr();

    let thread = this.get_active_thread();

    let futex_private = this.eval_libc_i32("FUTEX_PRIVATE_FLAG")?;
    let futex_wait = this.eval_libc_i32("FUTEX_WAIT")?;
    let futex_wake = this.eval_libc_i32("FUTEX_WAKE")?;

    match op & !futex_private {
        op if op == futex_wait => {
            if args.len() < 5 {
                throw_ub_format!("incorrect number of arguments for FUTEX_WAIT syscall: got {}, expected at least 5", args.len());
            }
            let timeout = this.read_scalar(args[4])?.check_init()?;
            if !this.is_null(timeout)? {
                throw_ub_format!("miri does not support timeouts for futex operations");
            }
            // Check the pointer for alignment. Atomic operations are only available for fully aligned values.
            this.memory.check_ptr_access(addr.ptr.into(), Size::from_bytes(4), Align::from_bytes(4).unwrap())?;
            // Read an `i32` through the pointer, regardless of any wrapper types (e.g. `AtomicI32`).
            let futex_val = this.read_scalar(addr.offset(Size::ZERO, MemPlaceMeta::None, this.machine.layouts.i32, this)?.into())?.to_i32()?;
            if val == futex_val {
                this.block_thread(thread);
                this.futex_wait(futex_ptr, thread);
                this.write_scalar(Scalar::from_i32(0), dest)?;
            } else {
                let eagain = this.eval_libc("EAGAIN")?;
                this.set_last_error(eagain)?;
                this.write_scalar(Scalar::from_i32(-1), dest)?;
            }
        }
        op if op == futex_wake => {
            let mut n = 0;
            for _ in 0..val {
                if let Some(thread) = this.futex_wake(futex_ptr) {
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
