use rustc_abi::CanonAbi;
use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::callconv::FnAbi;

use crate::shims::sig::check_min_vararg_count;
use crate::shims::unix::env::EvalContextExt;
use crate::shims::unix::linux_like::eventfd::EvalContextExt as _;
use crate::shims::unix::linux_like::sync::futex;
use crate::*;

pub fn syscall<'tcx>(
    ecx: &mut MiriInterpCx<'tcx>,
    link_name: Symbol,
    abi: &FnAbi<'tcx, Ty<'tcx>>,
    args: &[OpTy<'tcx>],
    dest: &MPlaceTy<'tcx>,
) -> InterpResult<'tcx> {
    let ([op], varargs) = ecx.check_shim_sig_variadic_lenient(abi, CanonAbi::C, link_name, args)?;
    // The syscall variadic function is legal to call with more arguments than needed,
    // extra arguments are simply ignored. The important check is that when we use an
    // argument, we have to also check all arguments *before* it to ensure that they
    // have the right type.

    let sys_getrandom = ecx.eval_libc("SYS_getrandom").to_target_usize(ecx)?;
    let sys_futex = ecx.eval_libc("SYS_futex").to_target_usize(ecx)?;
    let sys_eventfd2 = ecx.eval_libc("SYS_eventfd2").to_target_usize(ecx)?;
    let sys_gettid = ecx.eval_libc("SYS_gettid").to_target_usize(ecx)?;

    match ecx.read_target_usize(op)? {
        // `libc::syscall(NR_GETRANDOM, buf.as_mut_ptr(), buf.len(), GRND_NONBLOCK)`
        // is called if a `HashMap` is created the regular way (e.g. HashMap<K, V>).
        num if num == sys_getrandom => {
            // Used by getrandom 0.1
            // The first argument is the syscall id, so skip over it.
            let [ptr, len, flags] = check_min_vararg_count("syscall(SYS_getrandom, ...)", varargs)?;

            let ptr = ecx.read_pointer(ptr)?;
            let len = ecx.read_target_usize(len)?;
            // The only supported flags are GRND_RANDOM and GRND_NONBLOCK,
            // neither of which have any effect on our current PRNG.
            // See <https://github.com/rust-lang/rust/pull/79196> for a discussion of argument sizes.
            let _flags = ecx.read_scalar(flags)?.to_i32()?;

            ecx.gen_random(ptr, len)?;
            ecx.write_scalar(Scalar::from_target_usize(len, ecx), dest)?;
        }
        // `futex` is used by some synchronization primitives.
        num if num == sys_futex => {
            futex(ecx, varargs, dest)?;
        }
        num if num == sys_eventfd2 => {
            let [initval, flags] = check_min_vararg_count("syscall(SYS_evetfd2, ...)", varargs)?;

            let result = ecx.eventfd(initval, flags)?;
            ecx.write_int(result.to_i32()?, dest)?;
        }
        num if num == sys_gettid => {
            let result = ecx.unix_gettid("SYS_gettid")?;
            ecx.write_int(result.to_u32()?, dest)?;
        }
        num => {
            throw_unsup_format!("syscall: unsupported syscall number {num}");
        }
    };

    interp_ok(())
}
