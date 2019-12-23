use std::time::{Duration, SystemTime};

use crate::stacked_borrows::Tag;
use crate::*;
use helpers::immty_from_int_checked;

// Returns the time elapsed between now and the unix epoch as a `Duration`.
fn get_time<'tcx>() -> InterpResult<'tcx, Duration> {
    system_time_to_duration(&SystemTime::now())
}

/// Returns the time elapsed between the provided time and the unix epoch as a `Duration`.
pub fn system_time_to_duration<'tcx>(time: &SystemTime) -> InterpResult<'tcx, Duration> {
    time.duration_since(SystemTime::UNIX_EPOCH)
        .map_err(|_| err_unsup_format!("Times before the Unix epoch are not supported").into())
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
            immty_from_int_checked(tv_sec, this.libc_ty_layout("time_t")?)?,
            immty_from_int_checked(tv_nsec, this.libc_ty_layout("c_long")?)?,
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
            immty_from_int_checked(tv_sec, this.libc_ty_layout("time_t")?)?,
            immty_from_int_checked(tv_usec, this.libc_ty_layout("suseconds_t")?)?,
        ];

        this.write_packed_immediates(&tv, &imms)?;

        Ok(0)
    }
}
