use crate::*;

impl<'mir, 'tcx> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn emulate_foreign_item_by_name(
        &mut self,
        link_name: &str,
        args: &[OpTy<'tcx, Tag>],
        dest: PlaceTy<'tcx, Tag>,
    ) -> InterpResult<'tcx> {
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

            _ => throw_unsup_format!("can't call foreign function: {}", link_name),
        };

        Ok(())
    }
}

