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

            _ => throw_unsup_format!("can't call foreign function: {}", link_name),
        };

        Ok(())
    }
}
