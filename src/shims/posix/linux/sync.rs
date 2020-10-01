use crate::*;
use rustc_target::abi::{Align, Size};

/// Implementation of the SYS_futex syscall.
pub fn futex<'tcx>(
    this: &mut MiriEvalContext<'_, 'tcx>,
    args: &[OpTy<'tcx, Tag>],
    dest: PlaceTy<'tcx, Tag>,
) -> InterpResult<'tcx> {
    if args.len() < 4 {
        throw_ub_format!("incorrect number of arguments for futex syscall: got {}, expected at least 4", args.len());
    }
    let addr = this.read_scalar(args[1])?.check_init()?;
    let op = this.read_scalar(args[2])?.to_i32()?;
    let val = this.read_scalar(args[3])?.to_i32()?;

    this.memory.check_ptr_access(addr, Size::from_bytes(4), Align::from_bytes(4).unwrap())?;

    let addr = addr.assert_ptr().erase_tag();

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
            let futex_val = this.read_scalar_at_offset(args[1], 0, this.machine.layouts.i32)?.to_i32()?;
            if val == futex_val {
                this.block_thread(thread);
                this.futex_wait(addr, thread);
            } else {
                let eagain = this.eval_libc("EAGAIN")?;
                this.set_last_error(eagain)?;
            }
        }
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
