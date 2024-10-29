use rustc_span::Symbol;
use rustc_target::spec::abi::Abi;

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
        abi: Abi,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();
        match link_name.as_str() {
            // Threading
            "pthread_set_name_np" => {
                let [thread, name] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let max_len = usize::MAX; // FreeBSD does not seem to have a limit.
                // FreeBSD's pthread_set_name_np does not return anything.
                this.pthread_setname_np(
                    this.read_scalar(thread)?,
                    this.read_scalar(name)?,
                    max_len,
                    /* truncate */ false,
                )?;
            }
            "pthread_get_name_np" => {
                let [thread, name, len] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                // FreeBSD's pthread_get_name_np does not return anything
                // and uses strlcpy, which truncates the resulting value,
                // but always adds a null terminator (except for zero-sized buffers).
                // https://github.com/freebsd/freebsd-src/blob/c2d93a803acef634bd0eede6673aeea59e90c277/lib/libthr/thread/thr_info.c#L119-L144
                this.pthread_getname_np(
                    this.read_scalar(thread)?,
                    this.read_scalar(name)?,
                    this.read_scalar(len)?,
                    /* truncate */ true,
                )?;
            }

            // File related shims
            // For those, we both intercept `func` and `call@FBSD_1.0` symbols cases
            // since freebsd 12 the former form can be expected.
            "stat" | "stat@FBSD_1.0" => {
                let [path, buf] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let result = this.macos_fbsd_stat(path, buf)?;
                this.write_scalar(result, dest)?;
            }
            "lstat" | "lstat@FBSD_1.0" => {
                let [path, buf] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let result = this.macos_fbsd_lstat(path, buf)?;
                this.write_scalar(result, dest)?;
            }
            "fstat" | "fstat@FBSD_1.0" => {
                let [fd, buf] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let result = this.macos_fbsd_fstat(fd, buf)?;
                this.write_scalar(result, dest)?;
            }
            "readdir_r" | "readdir_r@FBSD_1.0" => {
                let [dirp, entry, result] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let result = this.macos_fbsd_readdir_r(dirp, entry, result)?;
                this.write_scalar(result, dest)?;
            }

            // Miscellaneous
            "__error" => {
                let [] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let errno_place = this.last_error_place()?;
                this.write_scalar(errno_place.to_ref(this).to_scalar(), dest)?;
            }

            // Incomplete shims that we "stub out" just to get pre-main initialization code to work.
            // These shims are enabled only when the caller is in the standard library.
            "pthread_attr_get_np" if this.frame_in_std() => {
                let [_thread, _attr] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                this.write_null(dest)?;
            }

            _ => return interp_ok(EmulateItemResult::NotSupported),
        }
        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
