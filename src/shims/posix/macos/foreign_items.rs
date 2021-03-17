use rustc_middle::mir;
use rustc_target::spec::abi::Abi;

use crate::*;
use helpers::{check_abi, check_arg_count};
use shims::posix::fs::EvalContextExt as _;
use shims::posix::thread::EvalContextExt as _;

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn emulate_foreign_item_by_name(
        &mut self,
        link_name: &str,
        abi: Abi,
        args: &[OpTy<'tcx, Tag>],
        dest: &PlaceTy<'tcx, Tag>,
        _ret: mir::BasicBlock,
    ) -> InterpResult<'tcx, bool> {
        let this = self.eval_context_mut();

        match link_name {
            // errno
            "__error" => {
                check_abi(abi, Abi::C { unwind: false })?;
                let &[] = check_arg_count(args)?;
                let errno_place = this.last_error_place()?;
                this.write_scalar(errno_place.to_ref().to_scalar()?, dest)?;
            }

            // File related shims
            "close" | "close$NOCANCEL" => {
                check_abi(abi, Abi::C { unwind: false })?;
                let &[ref result] = check_arg_count(args)?;
                let result = this.close(result)?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "stat" | "stat$INODE64" => {
                check_abi(abi, Abi::C { unwind: false })?;
                let &[ref path, ref buf] = check_arg_count(args)?;
                let result = this.macos_stat(path, buf)?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "lstat" | "lstat$INODE64" => {
                check_abi(abi, Abi::C { unwind: false })?;
                let &[ref path, ref buf] = check_arg_count(args)?;
                let result = this.macos_lstat(path, buf)?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "fstat" | "fstat$INODE64" => {
                check_abi(abi, Abi::C { unwind: false })?;
                let &[ref fd, ref buf] = check_arg_count(args)?;
                let result = this.macos_fstat(fd, buf)?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "opendir" | "opendir$INODE64" => {
                check_abi(abi, Abi::C { unwind: false })?;
                let &[ref name] = check_arg_count(args)?;
                let result = this.opendir(name)?;
                this.write_scalar(result, dest)?;
            }
            "readdir_r" | "readdir_r$INODE64" => {
                check_abi(abi, Abi::C { unwind: false })?;
                let &[ref dirp, ref entry, ref result] = check_arg_count(args)?;
                let result = this.macos_readdir_r(dirp, entry, result)?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "ftruncate" => {
                check_abi(abi, Abi::C { unwind: false })?;
                let &[ref fd, ref length] = check_arg_count(args)?;
                let result = this.ftruncate64(fd, length)?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }

            // Environment related shims
            "_NSGetEnviron" => {
                check_abi(abi, Abi::C { unwind: false })?;
                let &[] = check_arg_count(args)?;
                this.write_scalar(this.machine.env_vars.environ.unwrap().ptr, dest)?;
            }

            // Time related shims
            "gettimeofday" => {
                check_abi(abi, Abi::C { unwind: false })?;
                let &[ref tv, ref tz] = check_arg_count(args)?;
                let result = this.gettimeofday(tv, tz)?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "mach_absolute_time" => {
                check_abi(abi, Abi::C { unwind: false })?;
                let &[] = check_arg_count(args)?;
                let result = this.mach_absolute_time()?;
                this.write_scalar(Scalar::from_u64(result), dest)?;
            }

            "mach_timebase_info" => {
                check_abi(abi, Abi::C { unwind: false })?;
                let &[ref info] = check_arg_count(args)?;
                let result = this.mach_timebase_info(info)?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            },

            // Access to command-line arguments
            "_NSGetArgc" => {
                check_abi(abi, Abi::C { unwind: false })?;
                let &[] = check_arg_count(args)?;
                this.write_scalar(this.machine.argc.expect("machine must be initialized"), dest)?;
            }
            "_NSGetArgv" => {
                check_abi(abi, Abi::C { unwind: false })?;
                let &[] = check_arg_count(args)?;
                this.write_scalar(this.machine.argv.expect("machine must be initialized"), dest)?;
            }

            // Thread-local storage
            "_tlv_atexit" => {
                check_abi(abi, Abi::C { unwind: false })?;
                let &[ref dtor, ref data] = check_arg_count(args)?;
                let dtor = this.read_scalar(dtor)?.check_init()?;
                let dtor = this.memory.get_fn(dtor)?.as_instance()?;
                let data = this.read_scalar(data)?.check_init()?;
                let active_thread = this.get_active_thread();
                this.machine.tls.set_macos_thread_dtor(active_thread, dtor, data)?;
            }

            // Querying system information
            "pthread_get_stackaddr_np" => {
                check_abi(abi, Abi::C { unwind: false })?;
                let &[ref thread] = check_arg_count(args)?;
                this.read_scalar(thread)?.to_machine_usize(this)?;
                let stack_addr = Scalar::from_uint(STACK_ADDR, this.pointer_size());
                this.write_scalar(stack_addr, dest)?;
            }
            "pthread_get_stacksize_np" => {
                check_abi(abi, Abi::C { unwind: false })?;
                let &[ref thread] = check_arg_count(args)?;
                this.read_scalar(thread)?.to_machine_usize(this)?;
                let stack_size = Scalar::from_uint(STACK_SIZE, this.pointer_size());
                this.write_scalar(stack_size, dest)?;
            }

            // Threading
            "pthread_setname_np" => {
                check_abi(abi, Abi::C { unwind: false })?;
                let &[ref name] = check_arg_count(args)?;
                let name = this.read_scalar(name)?.check_init()?;
                this.pthread_setname_np(name)?;
            }

            // Incomplete shims that we "stub out" just to get pre-main initialization code to work.
            // These shims are enabled only when the caller is in the standard library.
            "mmap" if this.frame().instance.to_string().starts_with("std::sys::unix::") => {
                check_abi(abi, Abi::C { unwind: false })?;
                // This is a horrible hack, but since the guard page mechanism calls mmap and expects a particular return value, we just give it that value.
                let &[ref addr, _, _, _, _, _] = check_arg_count(args)?;
                let addr = this.read_scalar(addr)?.check_init()?;
                this.write_scalar(addr, dest)?;
            }

            _ => throw_unsup_format!("can't call foreign function: {}", link_name),
        };

        Ok(true)
    }
}

