use rustc_span::Symbol;
use rustc_target::spec::abi::Abi;

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
        abi: Abi,
        args: &[OpTy<'tcx>],
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx, EmulateItemResult> {
        let this = self.eval_context_mut();

        // See `fn emulate_foreign_item_inner` in `shims/foreign_items.rs` for the general pattern.

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
                let result = this.macos_fbsd_stat(path, buf)?;
                this.write_scalar(result, dest)?;
            }
            "lstat" | "lstat64" | "lstat$INODE64" => {
                let [path, buf] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let result = this.macos_fbsd_lstat(path, buf)?;
                this.write_scalar(result, dest)?;
            }
            "fstat" | "fstat64" | "fstat$INODE64" => {
                let [fd, buf] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let result = this.macos_fbsd_fstat(fd, buf)?;
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
                let result = this.macos_fbsd_readdir_r(dirp, entry, result)?;
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
                let environ = this.machine.env_vars.unix().environ();
                this.write_pointer(environ, dest)?;
            }

            // Random data generation
            "CCRandomGenerateBytes" => {
                let [bytes, count] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                let bytes = this.read_pointer(bytes)?;
                let count = this.read_target_usize(count)?;
                let success = this.eval_libc_i32("kCCSuccess");
                this.gen_random(bytes, count)?;
                this.write_int(success, dest)?;
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
                this.write_pointer(this.machine.argc.expect("machine must be initialized"), dest)?;
            }
            "_NSGetArgv" => {
                let [] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                this.write_pointer(this.machine.argv.expect("machine must be initialized"), dest)?;
            }
            "_NSGetExecutablePath" => {
                let [buf, bufsize] =
                    this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                this.check_no_isolation("`_NSGetExecutablePath`")?;

                let buf_ptr = this.read_pointer(buf)?;
                let bufsize = this.deref_pointer(bufsize)?;

                // Using the host current_exe is a bit off, but consistent with Linux
                // (where stdlib reads /proc/self/exe).
                let path = std::env::current_exe().unwrap();
                let (written, size_needed) = this.write_path_to_c_str(
                    &path,
                    buf_ptr,
                    this.read_scalar(&bufsize)?.to_u32()?.into(),
                )?;

                if written {
                    this.write_null(dest)?;
                } else {
                    this.write_scalar(Scalar::from_u32(size_needed.try_into().unwrap()), &bufsize)?;
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
                let active_thread = this.active_thread();
                this.machine.tls.add_macos_thread_dtor(active_thread, dtor, data)?;
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

            "os_unfair_lock_lock" => {
                let [lock_op] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                this.os_unfair_lock_lock(lock_op)?;
            }
            "os_unfair_lock_trylock" => {
                let [lock_op] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                this.os_unfair_lock_trylock(lock_op, dest)?;
            }
            "os_unfair_lock_unlock" => {
                let [lock_op] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                this.os_unfair_lock_unlock(lock_op)?;
            }
            "os_unfair_lock_assert_owner" => {
                let [lock_op] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                this.os_unfair_lock_assert_owner(lock_op)?;
            }
            "os_unfair_lock_assert_not_owner" => {
                let [lock_op] = this.check_shim(abi, Abi::C { unwind: false }, link_name, args)?;
                this.os_unfair_lock_assert_not_owner(lock_op)?;
            }

            _ => return interp_ok(EmulateItemResult::NotSupported),
        };

        interp_ok(EmulateItemResult::NeedsReturn)
    }
}
