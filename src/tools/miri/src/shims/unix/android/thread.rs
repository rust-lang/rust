use rustc_abi::Size;
use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::callconv::{Conv, FnAbi};

use crate::helpers::check_min_vararg_count;
use crate::shims::unix::thread::{EvalContextExt as _, ThreadNameResult};
use crate::*;

const TASK_COMM_LEN: u64 = 16;

pub fn prctl<'tcx>(
    ecx: &mut MiriInterpCx<'tcx>,
    link_name: Symbol,
    abi: &FnAbi<'tcx, Ty<'tcx>>,
    args: &[OpTy<'tcx>],
    dest: &MPlaceTy<'tcx>,
) -> InterpResult<'tcx> {
    let ([op], varargs) = ecx.check_shim_variadic(abi, Conv::C, link_name, args)?;

    // FIXME: Use constants once https://github.com/rust-lang/libc/pull/3941 backported to the 0.2 branch.
    let pr_set_name = 15;
    let pr_get_name = 16;

    let res = match ecx.read_scalar(op)?.to_i32()? {
        op if op == pr_set_name => {
            let [name] = check_min_vararg_count("prctl(PR_SET_NAME, ...)", varargs)?;
            let name = ecx.read_scalar(name)?;
            let thread = ecx.pthread_self()?;
            // The Linux kernel silently truncates long names.
            // https://www.man7.org/linux/man-pages/man2/PR_SET_NAME.2const.html
            let res =
                ecx.pthread_setname_np(thread, name, TASK_COMM_LEN, /* truncate */ true)?;
            assert_eq!(res, ThreadNameResult::Ok);
            Scalar::from_u32(0)
        }
        op if op == pr_get_name => {
            let [name] = check_min_vararg_count("prctl(PR_GET_NAME, ...)", varargs)?;
            let name = ecx.read_scalar(name)?;
            let thread = ecx.pthread_self()?;
            let len = Scalar::from_target_usize(TASK_COMM_LEN, ecx);
            ecx.check_ptr_access(
                name.to_pointer(ecx)?,
                Size::from_bytes(TASK_COMM_LEN),
                CheckInAllocMsg::MemoryAccess,
            )?;
            let res = ecx.pthread_getname_np(thread, name, len, /* truncate*/ false)?;
            assert_eq!(res, ThreadNameResult::Ok);
            Scalar::from_u32(0)
        }
        op => throw_unsup_format!("Miri does not support `prctl` syscall with op={}", op),
    };
    ecx.write_scalar(res, dest)?;
    interp_ok(())
}
