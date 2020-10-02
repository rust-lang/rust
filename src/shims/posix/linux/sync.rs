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
    let addr = args[1];
    let addr_scalar = this.read_scalar(addr)?.check_init()?;
    let futex_ptr = this.force_ptr(addr_scalar)?.erase_tag();
    let op = this.read_scalar(args[2])?.to_i32()?;
    let val = this.read_scalar(args[3])?.to_i32()?;

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
            this.memory.check_ptr_access(addr_scalar, Size::from_bytes(4), Align::from_bytes(4).unwrap())?;
            let futex_val = this.read_scalar_at_offset(args[1], 0, this.machine.layouts.i32)?.to_i32()?;
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
