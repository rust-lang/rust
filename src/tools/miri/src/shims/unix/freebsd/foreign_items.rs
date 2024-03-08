use rustc_span::Symbol;
use rustc_target::spec::abi::Abi;

use crate::*;
use shims::foreign_items::EmulateForeignItemResult;
use shims::unix::fs::EvalContextExt as _;
use shims::unix::thread::EvalContextExt as _;

pub fn is_dyn_sym(_name: &str) -> bool {
    false
}

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    fn emulate_foreign_item_inner(
        &mut self,
        link_name: Symbol,
        abi: Abi,
        args: &[OpTy<'tcx, Provenance>],
        dest: &MPlaceTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, EmulateForeignItemResult> {
        let this = self.eval_context_mut();
        match link_name.as_str() {
            // Threading
            "pthread_attr_get_np" if this.frame_in_std() => {
                let [_thread, _attr] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                this.write_null(dest)?;
            }
            "pthread_set_name_np" => {
                let [thread, name] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let max_len = usize::MAX; // FreeBSD does not seem to have a limit.
                // FreeBSD's pthread_set_name_np does not return anything.
                this.pthread_setname_np(
                    this.read_scalar(thread)?,
                    this.read_scalar(name)?,
                    max_len,
                )?;
            }
            "pthread_get_name_np" => {
                let [thread, name, len] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                // FreeBSD's pthread_get_name_np does not return anything.
                this.pthread_getname_np(
                    this.read_scalar(thread)?,
                    this.read_scalar(name)?,
                    this.read_scalar(len)?,
                )?;
            }
            "getrandom" => {
                let [ptr, len, flags] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let ptr = this.read_pointer(ptr)?;
                let len = this.read_target_usize(len)?;
                let _flags = this.read_scalar(flags)?.to_i32()?;
                // flags on freebsd does not really matter
                // in practice, GRND_RANDOM does not particularly draw from /dev/random
                // since it is the same as to /dev/urandom.
                // GRND_INSECURE is only an alias of GRND_NONBLOCK, which
                // does not affect the RNG.
                // https://man.freebsd.org/cgi/man.cgi?query=getrandom&sektion=2&n=1
                this.gen_random(ptr, len)?;
                this.write_scalar(Scalar::from_target_usize(len, this), dest)?;
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

            // errno
            "__error" => {
                let [] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let errno_place = this.last_error_place()?;
                this.write_scalar(errno_place.to_ref(this).to_scalar(), dest)?;
            }

            _ => return Ok(EmulateForeignItemResult::NotSupported),
        }
        Ok(EmulateForeignItemResult::NeedsJumping)
    }
}
