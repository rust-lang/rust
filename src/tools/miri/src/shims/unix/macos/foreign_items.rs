use rustc_span::Symbol;
use rustc_target::spec::abi::Abi;

use crate::*;
use shims::foreign_items::EmulateByNameResult;
use shims::unix::fs::EvalContextExt as _;
use shims::unix::thread::EvalContextExt as _;

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    fn emulate_foreign_item_by_name(
        &mut self,
        link_name: Symbol,
        abi: Abi,
        args: &[OpTy<'tcx, Provenance>],
        dest: &PlaceTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, EmulateByNameResult<'mir, 'tcx>> {
        let this = self.eval_context_mut();

        // See `fn emulate_foreign_item_by_name` in `shims/foreign_items.rs` for the general pattern.

        match link_name.as_str() {
            // errno
            "__error" => {
                let [] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let errno_place = this.last_error_place()?;
                this.write_scalar(errno_place.to_ref(this).to_scalar(), dest)?;
            }

            // File related shims
            "close$NOCANCEL" => {
                let [result] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let result = this.close(result)?;
                this.write_scalar(result, dest)?;
            }
            "stat" | "stat64" | "stat$INODE64" => {
                let [path, buf] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let result = this.macos_stat(path, buf)?;
                this.write_scalar(result, dest)?;
            }
            "lstat" | "lstat64" | "lstat$INODE64" => {
                let [path, buf] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let result = this.macos_lstat(path, buf)?;
                this.write_scalar(result, dest)?;
            }
            "fstat" | "fstat64" | "fstat$INODE64" => {
                let [fd, buf] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let result = this.macos_fstat(fd, buf)?;
                this.write_scalar(result, dest)?;
            }
            "opendir$INODE64" => {
                let [name] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let result = this.opendir(name)?;
                this.write_scalar(result, dest)?;
            }
            "readdir_r" | "readdir_r$INODE64" => {
                let [dirp, entry, result] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let result = this.macos_readdir_r(dirp, entry, result)?;
                this.write_scalar(result, dest)?;
            }
            "lseek" => {
                let [fd, offset, whence] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                // macOS is 64bit-only, so this is lseek64
                let result = this.lseek64(fd, offset, whence)?;
                this.write_scalar(result, dest)?;
            }
            "ftruncate" => {
                let [fd, length] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                // macOS is 64bit-only, so this is ftruncate64
                let result = this.ftruncate64(fd, length)?;
                this.write_scalar(result, dest)?;
            }
            "realpath$DARWIN_EXTSN" => {
                let [path, resolved_path] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let result = this.realpath(path, resolved_path)?;
                this.write_scalar(result, dest)?;
            }

            // Environment related shims
            "_NSGetEnviron" => {
                let [] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                this.write_pointer(
                    this.machine.env_vars.environ.expect("machine must be initialized").ptr,
                    dest,
                )?;
            }

            // Time related shims
            "mach_absolute_time" => {
                let [] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let result = this.mach_absolute_time()?;
                this.write_scalar(result, dest)?;
            }

            "mach_timebase_info" => {
                let [info] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let result = this.mach_timebase_info(info)?;
                this.write_scalar(result, dest)?;
            }

            // Access to command-line arguments
            "_NSGetArgc" => {
                let [] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                this.write_pointer(
                    this.machine.argc.expect("machine must be initialized").ptr,
                    dest,
                )?;
            }
            "_NSGetArgv" => {
                let [] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                this.write_pointer(
                    this.machine.argv.expect("machine must be initialized").ptr,
                    dest,
                )?;
            }
            "_NSGetExecutablePath" => {
                let [buf, bufsize] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                this.check_no_isolation("`_NSGetExecutablePath`")?;

                let buf_ptr = this.read_pointer(buf)?;
                let bufsize = this.deref_operand(bufsize)?;

                // Using the host current_exe is a bit off, but consistent with Linux
                // (where stdlib reads /proc/self/exe).
                let path = std::env::current_exe().unwrap();
                let (written, size_needed) = this.write_path_to_c_str(
                    &path,
                    buf_ptr,
                    this.read_scalar(&bufsize.into())?.to_u32()?.into(),
                )?;

                if written {
                    this.write_null(dest)?;
                } else {
                    this.write_scalar(
                        Scalar::from_u32(size_needed.try_into().unwrap()),
                        &bufsize.into(),
                    )?;
                    this.write_int(-1, dest)?;
                }
            }

            // Thread-local storage
            "_tlv_atexit" => {
                let [dtor, data] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let dtor = this.read_pointer(dtor)?;
                let dtor = this.get_ptr_fn(dtor)?.as_instance()?;
                let data = this.read_scalar(data)?;
                let active_thread = this.get_active_thread();
                this.machine.tls.set_macos_thread_dtor(active_thread, dtor, data)?;
            }

            // Querying system information
            "pthread_get_stackaddr_np" => {
                let [thread] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                this.read_target_usize(thread)?;
                let stack_addr = Scalar::from_uint(this.machine.stack_addr, this.pointer_size());
                this.write_scalar(stack_addr, dest)?;
            }
            "pthread_get_stacksize_np" => {
                let [thread] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                this.read_target_usize(thread)?;
                let stack_size = Scalar::from_uint(this.machine.stack_size, this.pointer_size());
                this.write_scalar(stack_size, dest)?;
            }

            // Threading
            "pthread_setname_np" => {
                let [name] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let thread = this.pthread_self()?;
                let max_len = this.eval_libc("MAXTHREADNAMESIZE").to_target_usize(this)?;
                let res = this.pthread_setname_np(
                    thread,
                    this.read_scalar(name)?,
                    max_len.try_into().unwrap(),
                )?;
                // Contrary to the manpage, `pthread_setname_np` on macOS still
                // returns an integer indicating success.
                this.write_scalar(res, dest)?;
            }
            "pthread_getname_np" => {
                let [thread, name, len] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let res = this.pthread_getname_np(
                    this.read_scalar(thread)?,
                    this.read_scalar(name)?,
                    this.read_scalar(len)?,
                )?;
                this.write_scalar(res, dest)?;
            }

            // Incomplete shims that we "stub out" just to get pre-main initialization code to work.
            // These shims are enabled only when the caller is in the standard library.
            "mmap" if this.frame_in_std() => {
                // This is a horrible hack, but since the guard page mechanism calls mmap and expects a particular return value, we just give it that value.
                let [addr, _, _, _, _, _] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let addr = this.read_scalar(addr)?;
                this.write_scalar(addr, dest)?;
            }

            _ => return Ok(EmulateByNameResult::NotSupported),
        };

        Ok(EmulateByNameResult::NeedsJumping)
    }
}
