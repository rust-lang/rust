use rustc_span::Symbol;
use rustc_target::abi::Size;
use rustc_target::spec::abi::Abi;

use crate::helpers::check_min_arg_count;
use crate::shims::unix::thread::EvalContextExt as _;
use crate::*;

const TASK_COMM_LEN: usize = 16;

pub fn prctl<'tcx>(
    this: &mut MiriInterpCx<'tcx>,
    link_name: Symbol,
    abi: Abi,
    args: &[OpTy<'tcx>],
    dest: &MPlaceTy<'tcx>,
) -> InterpResult<'tcx> {
    // We do not use `check_shim` here because `prctl` is variadic. The argument
    // count is checked bellow.
    this.check_abi_and_shim_symbol_clash(abi, Abi::C { unwind: false }, link_name)?;

    // FIXME: Use constants once https://github.com/rust-lang/libc/pull/3941 backported to the 0.2 branch.
    let pr_set_name = 15;
    let pr_get_name = 16;

    let [op] = check_min_arg_count("prctl", args)?;
    let res = match this.read_scalar(op)?.to_i32()? {
        op if op == pr_set_name => {
            let [_, name] = check_min_arg_count("prctl(PR_SET_NAME, ...)", args)?;
            let name = this.read_scalar(name)?;
            let thread = this.pthread_self()?;
            // The Linux kernel silently truncates long names.
            // https://www.man7.org/linux/man-pages/man2/PR_SET_NAME.2const.html
            let res =
                this.pthread_setname_np(thread, name, TASK_COMM_LEN, /* truncate */ true)?;
            assert!(res);
            Scalar::from_u32(0)
        }
        op if op == pr_get_name => {
            let [_, name] = check_min_arg_count("prctl(PR_GET_NAME, ...)", args)?;
            let name = this.read_scalar(name)?;
            let thread = this.pthread_self()?;
            let len = Scalar::from_target_usize(TASK_COMM_LEN as u64, this);
            this.check_ptr_access(
                name.to_pointer(this)?,
                Size::from_bytes(TASK_COMM_LEN),
                CheckInAllocMsg::MemoryAccessTest,
            )?;
            let res = this.pthread_getname_np(thread, name, len, /* truncate*/ false)?;
            assert!(res);
            Scalar::from_u32(0)
        }
        op => throw_unsup_format!("Miri does not support `prctl` syscall with op={}", op),
    };
    this.write_scalar(res, dest)?;
    interp_ok(())
}
