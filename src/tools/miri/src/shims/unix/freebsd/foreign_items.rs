use rustc_middle::ty::Ty;
use rustc_span::Symbol;
use rustc_target::callconv::{Conv, FnAbi};

use super::sync::EvalContextExt as _;
use crate::shims::unix::*;
use crate::*;

pub fn is_dyn_sym(_name: &str) -> bool {
    false
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn emulate_foreign_item_inner(
        &mut self,
        link_name: Symbol,
        abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();
        match link_name.as_str() {
            // Threading
            "pthread_setname_np" => {
                let [thread, name] = this.check_shim(abi, Conv::C, link_name, args)?;
                let max_len = usize::MAX; // FreeBSD does not seem to have a limit.
                let res = match this.pthread_setname_np(
                    this.read_scalar(thread)?,
                    this.read_scalar(name)?,
                    max_len,
                    /* truncate */ false,
                )? {
                    ThreadNameResult::Ok => Scalar::from_u32(0),
                    ThreadNameResult::NameTooLong => unreachable!(),
                    ThreadNameResult::ThreadNotFound => this.eval_libc("ESRCH"),
                };
                this.write_scalar(res, dest)?;
            }
            "pthread_getname_np" => {
                let [thread, name, len] = this.check_shim(abi, Conv::C, link_name, args)?;
                // FreeBSD's pthread_getname_np uses strlcpy, which truncates the resulting value,
                // but always adds a null terminator (except for zero-sized buffers).
                // https://github.com/freebsd/freebsd-src/blob/c2d93a803acef634bd0eede6673aeea59e90c277/lib/libthr/thread/thr_info.c#L119-L144
                let res = match this.pthread_getname_np(
                    this.read_scalar(thread)?,
                    this.read_scalar(name)?,
                    this.read_scalar(len)?,
                    /* truncate */ true,
                )? {
                    ThreadNameResult::Ok => Scalar::from_u32(0),
                    // `NameTooLong` is possible when the buffer is zero sized,
                    ThreadNameResult::NameTooLong => Scalar::from_u32(0),
                    ThreadNameResult::ThreadNotFound => this.eval_libc("ESRCH"),
                };
                this.write_scalar(res, dest)?;
            }

            // Synchronization primitives
            "_umtx_op" => {
                let [obj, op, val, uaddr, uaddr2] =
                    this.check_shim(abi, Conv::C, link_name, args)?;
                this._umtx_op(obj, op, val, uaddr, uaddr2, dest)?;
            }

            // File related shims
            // For those, we both intercept `func` and `call@FBSD_1.0` symbols cases
            // since freebsd 12 the former form can be expected.
            "stat" | "stat@FBSD_1.0" => {
                let [path, buf] = this.check_shim(abi, Conv::C, link_name, args)?;
                let result = this.macos_fbsd_solarish_stat(path, buf)?;
                this.write_scalar(result, dest)?;
            }
            "lstat" | "lstat@FBSD_1.0" => {
                let [path, buf] = this.check_shim(abi, Conv::C, link_name, args)?;
                let result = this.macos_fbsd_solarish_lstat(path, buf)?;
                this.write_scalar(result, dest)?;
            }
            "fstat" | "fstat@FBSD_1.0" => {
                let [fd, buf] = this.check_shim(abi, Conv::C, link_name, args)?;
                let result = this.macos_fbsd_solarish_fstat(fd, buf)?;
                this.write_scalar(result, dest)?;
            }
            "readdir_r" | "readdir_r@FBSD_1.0" => {
                let [dirp, entry, result] = this.check_shim(abi, Conv::C, link_name, args)?;
                let result = this.macos_fbsd_readdir_r(dirp, entry, result)?;
                this.write_scalar(result, dest)?;
            }

            // Miscellaneous
            "__error" => {
                let [] = this.check_shim(abi, Conv::C, link_name, args)?;
                let errno_place = this.last_error_place()?;
                this.write_scalar(errno_place.to_ref(this).to_scalar(), dest)?;
            }

            // Incomplete shims that we "stub out" just to get pre-main initialization code to work.
            // These shims are enabled only when the caller is in the standard library.
            "pthread_attr_get_np" if this.frame_in_std() => {
                let [_thread, _attr] = this.check_shim(abi, Conv::C, link_name, args)?;
                this.write_null(dest)?;
            }

            _ => return interp_ok(EmulateItemResult::NotSupported),
        }
        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
