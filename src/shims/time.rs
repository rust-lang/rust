use std::convert::TryFrom;
use std::time::{Duration, Instant, SystemTime};

use crate::*;
use thread::Time;

/// Returns the time elapsed between the provided time and the unix epoch as a `Duration`.
pub fn system_time_to_duration<'tcx>(time: &SystemTime) -> InterpResult<'tcx, Duration> {
    time.duration_since(SystemTime::UNIX_EPOCH)
        .map_err(|_| err_unsup_format!("times before the Unix epoch are not supported").into())
}

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriEvalContext<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriEvalContextExt<'mir, 'tcx> {
    fn clock_gettime(
        &mut self,
        clk_id_op: &OpTy<'tcx, Tag>,
        tp_op: &OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.assert_target_os("linux", "clock_gettime");
        this.check_no_isolation("`clock_gettime`")?;

        let clk_id = this.read_scalar(clk_id_op)?.to_i32()?;

        let duration = if clk_id == this.eval_libc_i32("CLOCK_REALTIME")? {
            system_time_to_duration(&SystemTime::now())?
        } else if clk_id == this.eval_libc_i32("CLOCK_MONOTONIC")? {
            // Absolute time does not matter, only relative time does, so we can just
            // use our own time anchor here.
            Instant::now().duration_since(this.machine.time_anchor)
        } else {
            let einval = this.eval_libc("EINVAL")?;
            this.set_last_error(einval)?;
            return Ok(-1);
        };

        let tv_sec = duration.as_secs();
        let tv_nsec = duration.subsec_nanos();

        this.write_int_fields(&[tv_sec.into(), tv_nsec.into()], &this.deref_operand(tp_op)?)?;

        Ok(0)
    }

    fn gettimeofday(
        &mut self,
        tv_op: &OpTy<'tcx, Tag>,
        tz_op: &OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.assert_target_os("macos", "gettimeofday");
        this.check_no_isolation("`gettimeofday`")?;

        // Using tz is obsolete and should always be null
        let tz = this.read_pointer(tz_op)?;
        if !this.ptr_is_null(tz)? {
            let einval = this.eval_libc("EINVAL")?;
            this.set_last_error(einval)?;
            return Ok(-1);
        }

        let duration = system_time_to_duration(&SystemTime::now())?;
        let tv_sec = duration.as_secs();
        let tv_usec = duration.subsec_micros();

        this.write_int_fields(&[tv_sec.into(), tv_usec.into()], &this.deref_operand(tv_op)?)?;

        Ok(0)
    }

    #[allow(non_snake_case)]
    fn GetSystemTimeAsFileTime(&mut self, LPFILETIME_op: &OpTy<'tcx, Tag>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        this.assert_target_os("windows", "GetSystemTimeAsFileTime");
        this.check_no_isolation("`GetSystemTimeAsFileTime`")?;

        let NANOS_PER_SEC = this.eval_windows_u64("time", "NANOS_PER_SEC")?;
        let INTERVALS_PER_SEC = this.eval_windows_u64("time", "INTERVALS_PER_SEC")?;
        let INTERVALS_TO_UNIX_EPOCH = this.eval_windows_u64("time", "INTERVALS_TO_UNIX_EPOCH")?;
        let NANOS_PER_INTERVAL = NANOS_PER_SEC / INTERVALS_PER_SEC;
        let SECONDS_TO_UNIX_EPOCH = INTERVALS_TO_UNIX_EPOCH / INTERVALS_PER_SEC;

        let duration = system_time_to_duration(&SystemTime::now())?
            + Duration::from_secs(SECONDS_TO_UNIX_EPOCH);
        let duration_ticks = u64::try_from(duration.as_nanos() / u128::from(NANOS_PER_INTERVAL))
            .map_err(|_| err_unsup_format!("programs running more than 2^64 Windows ticks after the Windows epoch are not supported"))?;

        let dwLowDateTime = u32::try_from(duration_ticks & 0x00000000FFFFFFFF).unwrap();
        let dwHighDateTime = u32::try_from((duration_ticks & 0xFFFFFFFF00000000) >> 32).unwrap();
        this.write_int_fields(
            &[dwLowDateTime.into(), dwHighDateTime.into()],
            &this.deref_operand(LPFILETIME_op)?,
        )?;

        Ok(())
    }

    #[allow(non_snake_case)]
    fn QueryPerformanceCounter(
        &mut self,
        lpPerformanceCount_op: &OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.assert_target_os("windows", "QueryPerformanceCounter");
        this.check_no_isolation("`QueryPerformanceCounter`")?;

        // QueryPerformanceCounter uses a hardware counter as its basis.
        // Miri will emulate a counter with a resolution of 1 nanosecond.
        let duration = Instant::now().duration_since(this.machine.time_anchor);
        let qpc = i64::try_from(duration.as_nanos()).map_err(|_| {
            err_unsup_format!("programs running longer than 2^63 nanoseconds are not supported")
        })?;
        this.write_scalar(
            Scalar::from_i64(qpc),
            &this.deref_operand(lpPerformanceCount_op)?.into(),
        )?;
        Ok(-1) // return non-zero on success
    }

    #[allow(non_snake_case)]
    fn QueryPerformanceFrequency(
        &mut self,
        lpFrequency_op: &OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.assert_target_os("windows", "QueryPerformanceFrequency");
        this.check_no_isolation("`QueryPerformanceFrequency`")?;

        // Retrieves the frequency of the hardware performance counter.
        // The frequency of the performance counter is fixed at system boot and
        // is consistent across all processors.
        // Miri emulates a "hardware" performance counter with a resolution of 1ns,
        // and thus 10^9 counts per second.
        this.write_scalar(
            Scalar::from_i64(1_000_000_000),
            &this.deref_operand(lpFrequency_op)?.into(),
        )?;
        Ok(-1) // Return non-zero on success
    }

    fn mach_absolute_time(&self) -> InterpResult<'tcx, u64> {
        let this = self.eval_context_ref();

        this.assert_target_os("macos", "mach_absolute_time");
        this.check_no_isolation("`mach_absolute_time`")?;

        // This returns a u64, with time units determined dynamically by `mach_timebase_info`.
        // We return plain nanoseconds.
        let duration = Instant::now().duration_since(this.machine.time_anchor);
        u64::try_from(duration.as_nanos()).map_err(|_| {
            err_unsup_format!("programs running longer than 2^64 nanoseconds are not supported")
                .into()
        })
    }

    fn mach_timebase_info(&mut self, info_op: &OpTy<'tcx, Tag>) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.assert_target_os("macos", "mach_timebase_info");
        this.check_no_isolation("`mach_timebase_info`")?;

        let info = this.deref_operand(info_op)?;

        // Since our emulated ticks in `mach_absolute_time` *are* nanoseconds,
        // no scaling needs to happen.
        let (numer, denom) = (1, 1);
        this.write_int_fields(&[numer.into(), denom.into()], &info)?;

        Ok(0) // KERN_SUCCESS
    }

    fn nanosleep(
        &mut self,
        req_op: &OpTy<'tcx, Tag>,
        _rem: &OpTy<'tcx, Tag>,
    ) -> InterpResult<'tcx, i32> {
        // Signal handlers are not supported, so rem will never be written to.

        let this = self.eval_context_mut();

        this.check_no_isolation("`nanosleep`")?;

        let duration = match this.read_timespec(&this.deref_operand(req_op)?)? {
            Some(duration) => duration,
            None => {
                let einval = this.eval_libc("EINVAL")?;
                this.set_last_error(einval)?;
                return Ok(-1);
            }
        };
        let timeout_time = Time::Monotonic(Instant::now().checked_add(duration).unwrap());

        let active_thread = this.get_active_thread();
        this.block_thread(active_thread);

        this.register_timeout_callback(
            active_thread,
            timeout_time,
            Box::new(move |ecx| {
                ecx.unblock_thread(active_thread);
                Ok(())
            }),
        );

        Ok(0)
    }
}
