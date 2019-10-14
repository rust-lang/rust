use std::time::{Duration, SystemTime};

use rustc::ty::layout::TyLayout;

use crate::stacked_borrows::Tag;
use crate::*;

// Returns the time elapsed between now and the unix epoch as a `Duration` and the sign of the time
// interval
fn get_time<'tcx>() -> InterpResult<'tcx, Duration> {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_err(|_| err_unsup_format!("Times before the Unix epoch are not supported").into())
}

fn int_to_immty_checked<'tcx>(
    int: i128,
    layout: TyLayout<'tcx>,
) -> InterpResult<'tcx, ImmTy<'tcx, Tag>> {
    // If `int` does not fit in `size` bits, we error instead of letting
    // `ImmTy::from_int` panic.
    let size = layout.size;
    let truncated = truncate(int as u128, size);
    if sign_extend(truncated, size) as i128 != int {
        throw_unsup_format!(
            "Signed value {:#x} does not fit in {} bits",
            int,
            size.bits()
        )
    }
    Ok(ImmTy::from_int(int, layout))
}

impl<'mir, 'tcx> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    // Foreign function used by linux
    fn clock_gettime(
        &mut self,
        clk_id_op: OpTy<'tcx, Tag>,
        tp_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.check_no_isolation("clock_gettime")?;

        let clk_id = this.read_scalar(clk_id_op)?.to_i32()?;
        if clk_id != this.eval_libc_i32("CLOCK_REALTIME")? {
            let einval = this.eval_libc("EINVAL")?;
            this.set_last_error(einval)?;
            return Ok(-1);
        }

        let tp = this.deref_operand(tp_op)?;

        let duration = get_time()?;
        let tv_sec = duration.as_secs() as i128;
        let tv_nsec = duration.subsec_nanos() as i128;

        let imms = [
            int_to_immty_checked(tv_sec, this.libc_ty_layout("time_t")?)?,
            int_to_immty_checked(tv_nsec, this.libc_ty_layout("c_long")?)?,
        ];

        this.write_packed_immediates(&tp, &imms)?;

        Ok(0)
    }
    // Foreign function used by generic unix (in particular macOS)
    fn gettimeofday(
        &mut self,
        tv_op: OpTy<'tcx, Tag>,
        tz_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.check_no_isolation("gettimeofday")?;
        // Using tz is obsolete and should always be null
        let tz = this.read_scalar(tz_op)?.not_undef()?;
        if !this.is_null(tz)? {
            let einval = this.eval_libc("EINVAL")?;
            this.set_last_error(einval)?;
            return Ok(-1);
        }

        let tv = this.deref_operand(tv_op)?;

        let duration = get_time()?;
        let tv_sec = duration.as_secs() as i128;
        let tv_usec = duration.subsec_micros() as i128;

        let imms = [
            int_to_immty_checked(tv_sec, this.libc_ty_layout("time_t")?)?,
            int_to_immty_checked(tv_usec, this.libc_ty_layout("suseconds_t")?)?,
        ];

        this.write_packed_immediates(&tv, &imms)?;

        Ok(0)
    }
}
