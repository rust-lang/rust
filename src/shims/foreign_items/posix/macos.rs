use crate::*;
use rustc::mir;

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
            "__error" => {
                let errno_place = this.machine.last_error.unwrap();
                this.write_scalar(errno_place.to_ref().to_scalar()?, dest)?;
            }

            // File related shims
             "open" => {
                let result = this.open(args[0], args[1])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            "close$NOCANCEL" => {
                let result = this.close(args[0])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            "stat$INODE64" => {
                let result = this.stat(args[0], args[1])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            "lstat$INODE64" => {
                let result = this.lstat(args[0], args[1])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            "fstat$INODE64" => {
                let result = this.fstat(args[0], args[1])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            "lseek" => {
                let result = this.lseek64(args[0], args[1], args[2])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            // Time related shims
            "gettimeofday" => {
                let result = this.gettimeofday(args[0], args[1])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            // macOS API stubs.
            "pthread_attr_get_np" => {
                this.write_null(dest)?;
            }

            "pthread_get_stackaddr_np" => {
                let stack_addr = Scalar::from_uint(STACK_ADDR, dest.layout.size);
                this.write_scalar(stack_addr, dest)?;
            }

            "pthread_get_stacksize_np" => {
                let stack_size = Scalar::from_uint(STACK_SIZE, dest.layout.size);
                this.write_scalar(stack_size, dest)?;
            }

            "_tlv_atexit" => {
                // FIXME: register the destructor.
            }

            "_NSGetArgc" => {
                this.write_scalar(this.machine.argc.expect("machine must be initialized"), dest)?;
            }

            "_NSGetArgv" => {
                this.write_scalar(this.machine.argv.expect("machine must be initialized"), dest)?;
            }

            "SecRandomCopyBytes" => {
                let len = this.read_scalar(args[1])?.to_machine_usize(this)?;
                let ptr = this.read_scalar(args[2])?.not_undef()?;
                this.gen_random(ptr, len as usize)?;
                this.write_null(dest)?;
            }

            "syscall" => {
                let sys_getrandom = this
                    .eval_path_scalar(&["libc", "SYS_getrandom"])?
                    .expect("Failed to get libc::SYS_getrandom")
                    .to_machine_usize(this)?;

                match this.read_scalar(args[0])?.to_machine_usize(this)? {
                    // `libc::syscall(NR_GETRANDOM, buf.as_mut_ptr(), buf.len(), GRND_NONBLOCK)`
                    // is called if a `HashMap` is created the regular way (e.g. HashMap<K, V>).
                    id if id == sys_getrandom => {
                        // The first argument is the syscall id,
                        // so skip over it.
                        super::getrandom(this, &args[1..], dest)?;
                    }
                    id => throw_unsup_format!("miri does not support syscall ID {}", id),
                }
            }

            _ => throw_unsup_format!("can't call foreign function: {}", link_name),
        };

        Ok(true)
    }
}

