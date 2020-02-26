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

            // The only reason this is not in the `posix` module is because the `linux` item has a
            // different name.
            "close$NOCANCEL" => {
                let result = this.close(args[0])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            "stat$INODE64" => {
                let result = this.macos_stat(args[0], args[1])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            "lstat$INODE64" => {
                let result = this.macos_lstat(args[0], args[1])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            "fstat$INODE64" => {
                let result = this.macos_fstat(args[0], args[1])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            // The only reason this is not in the `posix` module is because the `linux` item has a
            // different name.
            "opendir$INODE64" => {
                let result = this.opendir(args[0])?;
                this.write_scalar(result, dest)?;
            }

            // The `linux` module has a parallel foreign item, `readdir64_r`, which uses a
            // different struct layout.
            "readdir_r$INODE64" => {
                let result = this.macos_readdir_r(args[0], args[1], args[2])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            // Time related shims
            "gettimeofday" => {
                let result = this.gettimeofday(args[0], args[1])?;
                this.write_scalar(Scalar::from_int(result, dest.layout.size), dest)?;
            }

            // Other shims
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

            _ => throw_unsup_format!("can't call foreign function: {}", link_name),
        };

        Ok(true)
    }
}

