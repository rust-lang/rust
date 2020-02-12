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
            "__errno_location" => {
                let errno_place = this.machine.last_error.unwrap();
                this.write_scalar(errno_place.to_ref().to_scalar()?, dest)?;
            }

            // File related shims
            "open64" => {
                let result = this.open(args[0], args[1])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            "close" => {
                let result = this.close(args[0])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            "lseek64" => {
                let result = this.lseek64(args[0], args[1], args[2])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            // Time related shims
            "clock_gettime" => {
                let result = this.clock_gettime(args[0], args[1])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            "pthread_getattr_np" => {
                this.write_null(dest)?;
            }

            "syscall" => {
                let sys_getrandom = this
                    .eval_path_scalar(&["libc", "SYS_getrandom"])?
                    .expect("Failed to get libc::SYS_getrandom")
                    .to_machine_usize(this)?;

                let sys_statx = this
                    .eval_path_scalar(&["libc", "SYS_statx"])?
                    .expect("Failed to get libc::SYS_statx")
                    .to_machine_usize(this)?;

                match this.read_scalar(args[0])?.to_machine_usize(this)? {
                    // `libc::syscall(NR_GETRANDOM, buf.as_mut_ptr(), buf.len(), GRND_NONBLOCK)`
                    // is called if a `HashMap` is created the regular way (e.g. HashMap<K, V>).
                    id if id == sys_getrandom => {
                        // The first argument is the syscall id,
                        // so skip over it.
                        super::getrandom(this, &args[1..], dest)?;
                    }
                    id if id == sys_statx => {
                        // The first argument is the syscall id,
                        // so skip over it.
                        let result = this.statx(args[1], args[2], args[3], args[4], args[5])?;
                        this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
                    }
                    id => throw_unsup_format!("miri does not support syscall ID {}", id),
                }
            }

            "getrandom" => {
                super::getrandom(this, args, dest)?;
            }

            "sched_getaffinity" => {
                // Return an error; `num_cpus` then falls back to `sysconf`.
                this.write_scalar(Scalar::from_int(-1, dest.layout.size), dest)?;
            }

            _ => throw_unsup_format!("can't call foreign function: {}", link_name),
        };

        Ok(true)
    }
}
