use crate::*;
use rustc_middle::mir;

impl<'mir, 'tcx> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
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
                let errno_place = this.machine.last_error.unwrap();
                this.write_scalar(errno_place.to_ref().to_scalar()?, dest)?;
            }

            // File related shims
            "close$NOCANCEL" => {
                let result = this.close(args[0])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "stat$INODE64" => {
                let result = this.macos_stat(args[0], args[1])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "lstat$INODE64" => {
                let result = this.macos_lstat(args[0], args[1])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "fstat$INODE64" => {
                let result = this.macos_fstat(args[0], args[1])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "opendir$INODE64" => {
                let result = this.opendir(args[0])?;
                this.write_scalar(result, dest)?;
            }
            "readdir_r$INODE64" => {
                let result = this.macos_readdir_r(args[0], args[1], args[2])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }

            // Environment related shims
            "_NSGetEnviron" => {
                this.write_scalar(this.machine.env_vars.environ.unwrap().ptr, dest)?;
            }

            // Time related shims
            "gettimeofday" => {
                let result = this.gettimeofday(args[0], args[1])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "mach_absolute_time" => {
                let result = this.mach_absolute_time()?;
                this.write_scalar(Scalar::from_u64(result), dest)?;
            }

            "mach_timebase_info" => {
                let result = this.mach_timebase_info(args[0])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            },

            // Access to command-line arguments
            "_NSGetArgc" => {
                this.write_scalar(this.machine.argc.expect("machine must be initialized"), dest)?;
            }
            "_NSGetArgv" => {
                this.write_scalar(this.machine.argv.expect("machine must be initialized"), dest)?;
            }

            // Thread-local storage
            "_tlv_atexit" => {
                let dtor = this.read_scalar(args[0])?.not_undef()?;
                let dtor = this.memory.get_fn(dtor)?.as_instance()?;
                let data = this.read_scalar(args[1])?.not_undef()?;
                this.machine.tls.set_global_dtor(dtor, data)?;
            }

            // Querying system information
            "pthread_get_stackaddr_np" => {
                let _thread = this.read_scalar(args[0])?.not_undef()?;
                let stack_addr = Scalar::from_uint(STACK_ADDR, this.pointer_size());
                this.write_scalar(stack_addr, dest)?;
            }
            "pthread_get_stacksize_np" => {
                let _thread = this.read_scalar(args[0])?.not_undef()?;
                let stack_size = Scalar::from_uint(STACK_SIZE, this.pointer_size());
                this.write_scalar(stack_size, dest)?;
            }

            // Incomplete shims that we "stub out" just to get pre-main initialization code to work.
            // These shims are enabled only when the caller is in the standard library.
            "mmap" if this.frame().instance.to_string().starts_with("std::sys::unix::") => {
                // This is a horrible hack, but since the guard page mechanism calls mmap and expects a particular return value, we just give it that value.
                let addr = this.read_scalar(args[0])?.not_undef()?;
                this.write_scalar(addr, dest)?;
            }

            _ => throw_unsup_format!("can't call foreign function: {}", link_name),
        };

        Ok(true)
    }
}

