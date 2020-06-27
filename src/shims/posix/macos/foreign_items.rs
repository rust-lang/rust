use rustc_middle::mir;

use crate::*;
use helpers::check_arg_count;
use shims::posix::fs::EvalContextExt as _;
use shims::posix::thread::EvalContextExt as _;

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn emulate_foreign_item_by_name(
        &mut self,
        link_name: &str,
        args: &[OpTy<'tcx, Tag>],
        dest: PlaceTy<'tcx, Tag>,
        _ret: mir::BasicBlock,
    ) -> InterpResult<'tcx, bool> {
        let this = self.eval_context_mut();

        match link_name {
            // errno
            "__error" => {
                let &[] = check_arg_count(args)?;
                let errno_place = this.machine.last_error.unwrap();
                this.write_scalar(errno_place.to_ref().to_scalar()?, dest)?;
            }

            // File related shims
            "close$NOCANCEL" => {
                let &[result] = check_arg_count(args)?;
                let result = this.close(result)?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "stat$INODE64" => {
                let &[path, buf] = check_arg_count(args)?;
                let result = this.macos_stat(path, buf)?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "lstat$INODE64" => {
                let &[path, buf] = check_arg_count(args)?;
                let result = this.macos_lstat(path, buf)?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "fstat$INODE64" => {
                let &[fd, buf] = check_arg_count(args)?;
                let result = this.macos_fstat(fd, buf)?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "opendir$INODE64" => {
                let &[name] = check_arg_count(args)?;
                let result = this.opendir(name)?;
                this.write_scalar(result, dest)?;
            }
            "readdir_r$INODE64" => {
                let &[dirp, entry, result] = check_arg_count(args)?;
                let result = this.macos_readdir_r(dirp, entry, result)?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "ftruncate" => {
                let &[fd, length] = check_arg_count(args)?;
                let result = this.ftruncate64(fd, length)?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }

            // Environment related shims
            "_NSGetEnviron" => {
                let &[] = check_arg_count(args)?;
                this.write_scalar(this.machine.env_vars.environ.unwrap().ptr, dest)?;
            }

            // Time related shims
            "gettimeofday" => {
                let &[tv, tz] = check_arg_count(args)?;
                let result = this.gettimeofday(tv, tz)?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "mach_absolute_time" => {
                let &[] = check_arg_count(args)?;
                let result = this.mach_absolute_time()?;
                this.write_scalar(Scalar::from_u64(result), dest)?;
            }

            "mach_timebase_info" => {
                let &[info] = check_arg_count(args)?;
                let result = this.mach_timebase_info(info)?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            },

            // Access to command-line arguments
            "_NSGetArgc" => {
                let &[] = check_arg_count(args)?;
                this.write_scalar(this.machine.argc.expect("machine must be initialized"), dest)?;
            }
            "_NSGetArgv" => {
                let &[] = check_arg_count(args)?;
                this.write_scalar(this.machine.argv.expect("machine must be initialized"), dest)?;
            }

            // Thread-local storage
            "_tlv_atexit" => {
                let &[dtor, data] = check_arg_count(args)?;
                let dtor = this.read_scalar(dtor)?.not_undef()?;
                let dtor = this.memory.get_fn(dtor)?.as_instance()?;
                let data = this.read_scalar(data)?.not_undef()?;
                let active_thread = this.get_active_thread();
                this.machine.tls.set_macos_thread_dtor(active_thread, dtor, data)?;
            }

            // Querying system information
            "pthread_get_stackaddr_np" => {
                let &[thread] = check_arg_count(args)?;
                this.read_scalar(thread)?.to_machine_usize(this)?;
                let stack_addr = Scalar::from_uint(STACK_ADDR, this.pointer_size());
                this.write_scalar(stack_addr, dest)?;
            }
            "pthread_get_stacksize_np" => {
                let &[thread] = check_arg_count(args)?;
                this.read_scalar(thread)?.to_machine_usize(this)?;
                let stack_size = Scalar::from_uint(STACK_SIZE, this.pointer_size());
                this.write_scalar(stack_size, dest)?;
            }

            // Threading
            "pthread_setname_np" => {
                let &[name] = check_arg_count(args)?;
                let name = this.read_scalar(name)?.not_undef()?;
                this.pthread_setname_np(name)?;
            }

            // Incomplete shims that we "stub out" just to get pre-main initialization code to work.
            // These shims are enabled only when the caller is in the standard library.
            "mmap" if this.frame().instance.to_string().starts_with("std::sys::unix::") => {
                // This is a horrible hack, but since the guard page mechanism calls mmap and expects a particular return value, we just give it that value.
                let &[addr, _, _, _, _, _] = check_arg_count(args)?;
                let addr = this.read_scalar(addr)?.not_undef()?;
                this.write_scalar(addr, dest)?;
            }

            _ => throw_unsup_format!("can't call foreign function: {}", link_name),
        };

        Ok(true)
    }
}

