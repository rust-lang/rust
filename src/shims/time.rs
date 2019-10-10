use crate::stacked_borrows::Tag;
use crate::*;

use std::time::{Duration, SystemTime};

fn get_time() -> (Duration, i128) {
    let mut sign = 1;
    let duration = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_else(|e| {
            sign = -1;
            e.duration()
        });
    (duration, sign)
}

impl<'mir, 'tcx> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn clock_gettime(
        &mut self,
        clk_id_op: OpTy<'tcx, Tag>,
        tp_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        if !this.machine.communicate {
            throw_unsup_format!("`clock_gettime` not available when isolation is enabled")
        }

        let clk_id = this.read_scalar(clk_id_op)?.to_i32()?;
        if clk_id != this.eval_libc_i32("CLOCK_REALTIME")? {
            let einval = this.eval_libc("EINVAL")?;
            this.set_last_error(einval)?;
            return Ok(-1);
        }

        let tp = this.force_ptr(this.read_scalar(tp_op)?.not_undef()?)?;

        let (duration, sign) = get_time();
        let tv_sec = sign * (duration.as_secs() as i128);
        let tv_nsec = duration.subsec_nanos() as i128;
        this.write_c_ints(&tp, &[tv_sec, tv_nsec], &["time_t", "c_long"])?;

        Ok(0)
    }

    fn gettimeofday(
        &mut self,
        tv_op: OpTy<'tcx, Tag>,
        tz_op: OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        if !this.machine.communicate {
            throw_unsup_format!("`gettimeofday` not available when isolation is enabled")
        }
        // Using tz is obsolete and should always be null
        let tz = this.read_scalar(tz_op)?.not_undef()?;
        if !this.is_null(tz)? {
            let einval = this.eval_libc("EINVAL")?;
            this.set_last_error(einval)?;
            return Ok(-1);
        }

        let tv = this.force_ptr(this.read_scalar(tv_op)?.not_undef()?)?;

        let (duration, sign) = get_time();
        let tv_sec = sign * (duration.as_secs() as i128);
        let tv_usec = duration.subsec_micros() as i128;

        this.write_c_ints(&tv, &[tv_sec, tv_usec], &["time_t", "suseconds_t"])?;

        Ok(0)
    }
}
