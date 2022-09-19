use std::time::{Duration, SystemTime};

use crate::*;

/// Returns the time elapsed between the provided time and the unix epoch as a `Duration`.
pub fn system_time_to_duration<'tcx>(time: &SystemTime) -> InterpResult<'tcx, Duration> {
    time.duration_since(SystemTime::UNIX_EPOCH)
        .map_err(|_| err_unsup_format!("times before the Unix epoch are not supported").into())
}

impl<'mir, 'tcx: 'mir> EvalContextExt<'mir, 'tcx> for crate::MiriInterpCx<'mir, 'tcx> {}
pub trait EvalContextExt<'mir, 'tcx: 'mir>: crate::MiriInterpCxExt<'mir, 'tcx> {
    fn clock_gettime(
        &mut self,
        clk_id_op: &OpTy<'tcx, Provenance>,
        tp_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, Scalar<Provenance>> {
        // This clock support is deliberately minimal because a lot of clock types have fiddly
        // properties (is it possible for Miri to be suspended independently of the host?). If you
        // have a use for another clock type, please open an issue.

        let this = self.eval_context_mut();

        this.assert_target_os("linux", "clock_gettime");

        let clk_id = this.read_scalar(clk_id_op)?.to_i32()?;

        // Linux has two main kinds of clocks. REALTIME clocks return the actual time since the
        // Unix epoch, including effects which may cause time to move backwards such as NTP.
        // Linux further distinguishes regular and "coarse" clocks, but the "coarse" version
        // is just specified to be "faster and less precise", so we implement both the same way.
        let absolute_clocks =
            [this.eval_libc_i32("CLOCK_REALTIME")?, this.eval_libc_i32("CLOCK_REALTIME_COARSE")?];
        // The second kind is MONOTONIC clocks for which 0 is an arbitrary time point, but they are
        // never allowed to go backwards. We don't need to do any additonal monotonicity
        // enforcement because std::time::Instant already guarantees that it is monotonic.
        let relative_clocks =
            [this.eval_libc_i32("CLOCK_MONOTONIC")?, this.eval_libc_i32("CLOCK_MONOTONIC_COARSE")?];

        let duration = if absolute_clocks.contains(&clk_id) {
            this.check_no_isolation("`clock_gettime` with `REALTIME` clocks")?;
            system_time_to_duration(&SystemTime::now())?
        } else if relative_clocks.contains(&clk_id) {
            this.machine.clock.now().duration_since(this.machine.clock.anchor())
        } else {
            let einval = this.eval_libc("EINVAL")?;
            this.set_last_error(einval)?;
            return Ok(Scalar::from_i32(-1));
        };

        let tv_sec = duration.as_secs();
        let tv_nsec = duration.subsec_nanos();

        this.write_int_fields(&[tv_sec.into(), tv_nsec.into()], &this.deref_operand(tp_op)?)?;

        Ok(Scalar::from_i32(0))
    }

    fn gettimeofday(
        &mut self,
        tv_op: &OpTy<'tcx, Provenance>,
        tz_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.assert_target_os_is_unix("gettimeofday");
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

    #[allow(non_snake_case, clippy::integer_arithmetic)]
    fn GetSystemTimeAsFileTime(
        &mut self,
        LPFILETIME_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx> {
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
        lpPerformanceCount_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.assert_target_os("windows", "QueryPerformanceCounter");

        // QueryPerformanceCounter uses a hardware counter as its basis.
        // Miri will emulate a counter with a resolution of 1 nanosecond.
        let duration = this.machine.clock.now().duration_since(this.machine.clock.anchor());
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
        lpFrequency_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.assert_target_os("windows", "QueryPerformanceFrequency");

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

    fn mach_absolute_time(&self) -> InterpResult<'tcx, Scalar<Provenance>> {
        let this = self.eval_context_ref();

        this.assert_target_os("macos", "mach_absolute_time");

        // This returns a u64, with time units determined dynamically by `mach_timebase_info`.
        // We return plain nanoseconds.
        let duration = this.machine.clock.now().duration_since(this.machine.clock.anchor());
        let res = u64::try_from(duration.as_nanos()).map_err(|_| {
            err_unsup_format!("programs running longer than 2^64 nanoseconds are not supported")
        })?;
        Ok(Scalar::from_u64(res))
    }

    fn mach_timebase_info(
        &mut self,
        info_op: &OpTy<'tcx, Provenance>,
    ) -> InterpResult<'tcx, Scalar<Provenance>> {
        let this = self.eval_context_mut();

        this.assert_target_os("macos", "mach_timebase_info");

        let info = this.deref_operand(info_op)?;

        // Since our emulated ticks in `mach_absolute_time` *are* nanoseconds,
        // no scaling needs to happen.
        let (numer, denom) = (1, 1);
        this.write_int_fields(&[numer.into(), denom.into()], &info)?;

        Ok(Scalar::from_i32(0)) // KERN_SUCCESS
    }

    fn nanosleep(
        &mut self,
        req_op: &OpTy<'tcx, Provenance>,
        _rem: &OpTy<'tcx, Provenance>, // Signal handlers are not supported, so rem will never be written to.
    ) -> InterpResult<'tcx, i32> {
        let this = self.eval_context_mut();

        this.assert_target_os_is_unix("nanosleep");

        let duration = match this.read_timespec(&this.deref_operand(req_op)?)? {
            Some(duration) => duration,
            None => {
                let einval = this.eval_libc("EINVAL")?;
                this.set_last_error(einval)?;
                return Ok(-1);
            }
        };
        // If adding the duration overflows, let's just sleep for an hour. Waking up early is always acceptable.
        let now = this.machine.clock.now();
        let timeout_time = now
            .checked_add(duration)
            .unwrap_or_else(|| now.checked_add(Duration::from_secs(3600)).unwrap());

        let active_thread = this.get_active_thread();
        this.block_thread(active_thread);

        this.register_timeout_callback(
            active_thread,
            Time::Monotonic(timeout_time),
            Box::new(move |ecx| {
                ecx.unblock_thread(active_thread);
                Ok(())
            }),
        );

        Ok(0)
    }

    #[allow(non_snake_case)]
    fn Sleep(&mut self, timeout: &OpTy<'tcx, Provenance>) -> InterpResult<'tcx> {
        let this = self.eval_context_mut();

        this.assert_target_os("windows", "Sleep");

        let timeout_ms = this.read_scalar(timeout)?.to_u32()?;

        let duration = Duration::from_millis(timeout_ms.into());
        let timeout_time = this.machine.clock.now().checked_add(duration).unwrap();

        let active_thread = this.get_active_thread();
        this.block_thread(active_thread);

        this.register_timeout_callback(
            active_thread,
            Time::Monotonic(timeout_time),
            Box::new(move |ecx| {
                ecx.unblock_thread(active_thread);
                Ok(())
            }),
        );

        Ok(())
    }
}
