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
            "__errno_location" => {
                let errno_place = this.machine.last_error.unwrap();
                this.write_scalar(errno_place.to_ref().to_scalar()?, dest)?;
            }

            // File related shims (but also see "syscall" below for statx)
            // These symbols have different names on Linux and macOS, which is the only reason they are not
            // in the `posix` module.
            "close" => {
                let result = this.close(args[0])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            "opendir" => {
                let result = this.opendir(args[0])?;
                this.write_scalar(result, dest)?;
            }
            "readdir64_r" => {
                let result = this.linux_readdir64_r(args[0], args[1], args[2])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }
            // Linux-only
            "posix_fadvise" => {
                let _fd = this.read_scalar(args[0])?.to_i32()?;
                let _offset = this.read_scalar(args[1])?.to_machine_isize(this)?;
                let _len = this.read_scalar(args[2])?.to_machine_isize(this)?;
                let _advice = this.read_scalar(args[3])?.to_i32()?;
                // fadvise is only informational, we can ignore it.
                this.write_null(dest)?;
            }

            // Time related shims
            "clock_gettime" => {
                // This is a POSIX function but it has only been tested on linux.
                let result = this.clock_gettime(args[0], args[1])?;
                this.write_scalar(Scalar::from_i32(result), dest)?;
            }

            // Querying system information
            "pthread_attr_getstack" => {
                // We don't support "pthread_attr_setstack", so we just pretend all stacks have the same values here.
                let _attr_place = this.deref_operand(args[0])?;
                let addr_place = this.deref_operand(args[1])?;
                let size_place = this.deref_operand(args[2])?;

                this.write_scalar(
                    Scalar::from_uint(STACK_ADDR, this.pointer_size()),
                    addr_place.into(),
                )?;
                this.write_scalar(
                    Scalar::from_uint(STACK_SIZE, this.pointer_size()),
                    size_place.into(),
                )?;

                // Return success (`0`).
                this.write_null(dest)?;
            }

            // Dynamically invoked syscalls
            "syscall" => {
                let sys_getrandom = this
                    .eval_libc("SYS_getrandom")?
                    .to_machine_usize(this)?;

                let sys_statx = this
                    .eval_libc("SYS_statx")?
                    .to_machine_usize(this)?;

                match this.read_scalar(args[0])?.to_machine_usize(this)? {
                    // `libc::syscall(NR_GETRANDOM, buf.as_mut_ptr(), buf.len(), GRND_NONBLOCK)`
                    // is called if a `HashMap` is created the regular way (e.g. HashMap<K, V>).
                    id if id == sys_getrandom => {
                        // The first argument is the syscall id, so skip over it.
                        getrandom(this, &args[1..], dest)?;
                    }
                    // `statx` is used by `libstd` to retrieve metadata information on `linux`
                    // instead of using `stat`,`lstat` or `fstat` as on `macos`.
                    id if id == sys_statx => {
                        // The first argument is the syscall id, so skip over it.
                        let result = this.linux_statx(args[1], args[2], args[3], args[4], args[5])?;
                        this.write_scalar(Scalar::from_machine_isize(result.into(), this), dest)?;
                    }
                    id => throw_unsup_format!("miri does not support syscall ID {}", id),
                }
            }

            // Miscelanneous
            "getrandom" => {
                getrandom(this, args, dest)?;
            }
            "sched_getaffinity" => {
                let _pid = this.read_scalar(args[0])?.to_i32()?;
                let _cpusetsize = this.read_scalar(args[1])?.to_machine_usize(this)?;
                let _mask = this.deref_operand(args[2])?;
                // FIXME: we just return an error; `num_cpus` then falls back to `sysconf`.
                let einval = this.eval_libc("EINVAL")?;
                this.set_last_error(einval)?;
                this.write_scalar(Scalar::from_i32(-1), dest)?;
            }

            // Incomplete shims that we "stub out" just to get pre-main initialization code to work.
            // These shims are enabled only when the caller is in the standard library.
            "pthread_getattr_np" if this.frame().instance.to_string().starts_with("std::sys::unix::") => {
                this.write_null(dest)?;
            }

            _ => throw_unsup_format!("can't call foreign function: {}", link_name),
        };

        Ok(true)
    }
}

// Shims the linux `getrandom` syscall.
fn getrandom<'tcx>(
    this: &mut MiriEvalContext<'_, 'tcx>,
    args: &[OpTy<'tcx, Tag>],
    dest: PlaceTy<'tcx, Tag>,
) -> InterpResult<'tcx> {
    let ptr = this.read_scalar(args[0])?.not_undef()?;
    let len = this.read_scalar(args[1])?.to_machine_usize(this)?;

    // The only supported flags are GRND_RANDOM and GRND_NONBLOCK,
    // neither of which have any effect on our current PRNG.
    let _flags = this.read_scalar(args[2])?.to_i32()?;

    this.gen_random(ptr, len)?;
    this.write_scalar(Scalar::from_machine_usize(len, this), dest)?;
    Ok(())
}
