use std::ffi::OsStr;
use std::sync::atomic;
use std::time::{Duration, SystemTime};

use jiff::Timestamp;
use jiff::tz::TimeZone;
use rustc_target::spec::Os;

use crate::*;

/// Returns the time elapsed between the provided time and the unix epoch as a `Duration`.
pub fn system_time_to_duration<'tcx>(time: &SystemTime) -> InterpResult<'tcx, Duration> {
    time.duration_since(SystemTime::UNIX_EPOCH)
        .map_err(|_| err_unsup_format!("times before the Unix epoch are not supported"))
        .into()
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    fn parse_clockid(&self, clk_id: i32) -> Option<TimeoutClock> {
        // This clock support is deliberately minimal because a lot of clock types have fiddly
        // properties (is it possible for Miri to be suspended independently of the host?). If you
        // have a use for another clock type, please open an issue.
        let ecx = self.eval_context_ref();

        // Portable names that exist everywhere.
        if clk_id == ecx.eval_libc_i32("CLOCK_REALTIME") {
            return Some(TimeoutClock::RealTime);
        } else if clk_id == ecx.eval_libc_i32("CLOCK_MONOTONIC") {
            return Some(TimeoutClock::Monotonic);
        }

        // Some further platform-specific names we support.
        match &ecx.tcx.sess.target.os {
            Os::Linux | Os::FreeBsd | Os::Android => {
                // Linux further distinguishes regular and "coarse" clocks, but the "coarse" version
                // is just specified to be "faster and less precise", so we treat it like normal
                // clocks.
                if clk_id == ecx.eval_libc_i32("CLOCK_REALTIME_COARSE") {
                    return Some(TimeoutClock::RealTime);
                } else if clk_id == ecx.eval_libc_i32("CLOCK_MONOTONIC_COARSE") {
                    return Some(TimeoutClock::Monotonic);
                }
            }
            #[allow(clippy::collapsible_match)] // collapsing would remove symmetry
            Os::MacOs => {
                // `CLOCK_UPTIME_RAW` supposed to not increment while the system is asleep... but
                // that's not really something a program running inside Miri can tell, anyway.
                // We need to support it because std uses it.
                if clk_id == ecx.eval_libc_i32("CLOCK_UPTIME_RAW") {
                    return Some(TimeoutClock::Monotonic);
                }
            }
            _ => {}
        }

        None
    }

    fn clock_gettime(
        &mut self,
        clk_id_op: &OpTy<'tcx>,
        tp_op: &OpTy<'tcx>,
        dest: &MPlaceTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let ecx = self.eval_context_mut();

        ecx.assert_target_os_is_unix("clock_gettime");

        let clk_id = ecx.read_scalar(clk_id_op)?.to_i32()?;
        let tp = ecx.deref_pointer(tp_op)?;

        let duration = match ecx.parse_clockid(clk_id) {
            Some(TimeoutClock::RealTime) => {
                ecx.check_no_isolation("`clock_gettime` with `REALTIME` clocks")?;
                system_time_to_duration(&SystemTime::now())?
            }
            Some(TimeoutClock::Monotonic) =>
                ecx.machine
                    .monotonic_clock
                    .now()
                    .duration_since(ecx.machine.monotonic_clock.epoch()),
            None => {
                return ecx.set_errno_and_return_neg1(LibcError("EINVAL"), dest);
            }
        };

        let tv_sec = duration.as_secs();
        let tv_nsec = duration.subsec_nanos();

        ecx.write_int_fields(&[tv_sec.into(), tv_nsec.into()], &tp)?;
        ecx.write_int(0, dest)?;

        interp_ok(())
    }

    fn gettimeofday(
        &mut self,
        tv_op: &OpTy<'tcx>,
        tz_op: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let ecx = self.eval_context_mut();

        ecx.assert_target_os_is_unix("gettimeofday");
        ecx.check_no_isolation("`gettimeofday`")?;

        let tv = ecx.deref_pointer(tv_op)?;

        // Using tz is obsolete and should always be null
        let tz = ecx.read_pointer(tz_op)?;
        if !ecx.ptr_is_null(tz)? {
            throw_unsup_format!(
                "gettimeofday with non-null `tz` pointer is obsolete and not supported"
            );
        }

        let duration = system_time_to_duration(&SystemTime::now())?;
        let tv_sec = duration.as_secs();
        let tv_usec = duration.subsec_micros();

        ecx.write_int_fields(&[tv_sec.into(), tv_usec.into()], &tv)?;

        interp_ok(Scalar::from_i32(0))
    }

    // The localtime() function shall convert the time in seconds since the Epoch pointed to by
    // timer into a broken-down time, expressed as a local time.
    // https://linux.die.net/man/3/localtime_r
    fn localtime_r(
        &mut self,
        timep: &OpTy<'tcx>,
        result_op: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Pointer> {
        let ecx = self.eval_context_mut();

        ecx.assert_target_os_is_unix("localtime_r");
        ecx.check_no_isolation("`localtime_r`")?;

        let timep = ecx.deref_pointer(timep)?;
        let result = ecx.deref_pointer(result_op)?;

        // The input "represents the number of seconds elapsed since the Epoch,
        // 1970-01-01 00:00:00 +0000 (UTC)".
        let sec_since_epoch: i64 = ecx.read_int::<i64>(&timep)?;
        let timestamp = Timestamp::new(sec_since_epoch, 0).expect("timestamp should be valid");

        // Figure out what time zone is in use
        let tz = match ecx.get_env_var(OsStr::new("TZ"))? {
            Some(tz_name) => {
                tz_name.to_str().and_then(|tz| TimeZone::get(tz).ok()).unwrap_or_else(|| {
                    // jiff does not support this timezone. That also happens for obsolete timezones
                    // that typically work on POSIX systems, like "PST8PDT". So let's emit a warning
                    // to make the user aware of this possible discrepancy.
                    static DEDUP: atomic::AtomicBool = atomic::AtomicBool::new(false);
                    if !DEDUP.swap(true, atomic::Ordering::Relaxed) {
                        ecx.emit_diagnostic(NonHaltingDiagnostic::UnknownTimezone {
                            timezone: tz_name.to_string_lossy().into_owned(),
                        });
                    }

                    TimeZone::UTC
                })
            }
            None => {
                // Fall back to system timezone when TZ is not set. This reads the host's
                // "TZ", which is strictly speaking wrong, but close enough.
                TimeZone::system()
            }
        };

        // Put the two together. Looks like we have to do this twice to get all the information.
        let dt = timestamp.to_zoned_datetime(tz.clone());
        let offset_info = tz.to_offset_info(timestamp);

        ecx.write_int_fields_named(
            &[
                ("tm_sec", dt.second().into()),
                ("tm_min", dt.minute().into()),
                ("tm_hour", dt.hour().into()),
                ("tm_mday", dt.day().into()),
                ("tm_mon", dt.month().strict_sub(1).into()),
                ("tm_year", dt.year().strict_sub(1900).into()),
                ("tm_wday", dt.weekday().to_sunday_zero_offset().into()),
                ("tm_yday", dt.day_of_year().strict_sub(1).into()),
                ("tm_isdst", offset_info.dst().is_dst().into()),
            ],
            &result,
        )?;

        // solaris/illumos system tm struct does not have the additional tm_zone/tm_gmtoff fields.
        // https://docs.oracle.com/cd/E36784_01/html/E36874/localtime-r-3c.html
        if !matches!(&ecx.tcx.sess.target.os, Os::Solaris | Os::Illumos) {
            // Get relevant information.
            let offset_in_seconds = offset_info.offset().seconds();
            let timezone_name = offset_info.abbreviation();

            // Allocate the string with trailing null. This acts like a global variable:
            // it is deduplicated and the user is not expected to free it.
            let tm_zone = ecx.allocate_str_dedup(&format!("{timezone_name}\0"))?;

            // Write the timezone pointer and offset into the result structure.
            ecx.write_pointer(tm_zone.ptr(), &ecx.project_field_named(&result, "tm_zone")?)?;
            ecx.write_int_fields_named(&[("tm_gmtoff", offset_in_seconds.into())], &result)?;
        }
        interp_ok(result.ptr())
    }
    #[allow(non_snake_case, clippy::arithmetic_side_effects)]
    fn GetSystemTimeAsFileTime(
        &mut self,
        shim_name: &str,
        LPFILETIME_op: &OpTy<'tcx>,
    ) -> InterpResult<'tcx> {
        let ecx = self.eval_context_mut();

        ecx.assert_target_os(Os::Windows, shim_name);
        ecx.check_no_isolation(shim_name)?;

        let filetime = ecx.deref_pointer(LPFILETIME_op)?;

        let duration = ecx.system_time_since_windows_epoch(&SystemTime::now())?;
        let duration_ticks = ecx.windows_ticks_for(duration)?;

        let dwLowDateTime = u32::try_from(duration_ticks & 0x00000000FFFFFFFF).unwrap();
        let dwHighDateTime = u32::try_from((duration_ticks & 0xFFFFFFFF00000000) >> 32).unwrap();
        ecx.write_int_fields(&[dwLowDateTime.into(), dwHighDateTime.into()], &filetime)?;

        interp_ok(())
    }

    #[allow(non_snake_case)]
    fn QueryPerformanceCounter(
        &mut self,
        lpPerformanceCount_op: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let ecx = self.eval_context_mut();

        ecx.assert_target_os(Os::Windows, "QueryPerformanceCounter");

        // QueryPerformanceCounter uses a hardware counter as its basis.
        // Miri will emulate a counter with a resolution of 1 nanosecond.
        let duration =
            ecx.machine.monotonic_clock.now().duration_since(ecx.machine.monotonic_clock.epoch());
        let qpc = i64::try_from(duration.as_nanos()).map_err(|_| {
            err_unsup_format!("programs running longer than 2^63 nanoseconds are not supported")
        })?;

        ecx.write_scalar(Scalar::from_i64(qpc), &ecx.deref_pointer(lpPerformanceCount_op)?)?;
        interp_ok(Scalar::from_i32(-1)) // return non-zero on success
    }

    #[allow(non_snake_case)]
    fn QueryPerformanceFrequency(
        &mut self,
        lpFrequency_op: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let ecx = self.eval_context_mut();

        ecx.assert_target_os(Os::Windows, "QueryPerformanceFrequency");

        // Retrieves the frequency of the hardware performance counter.
        // The frequency of the performance counter is fixed at system boot and
        // is consistent across all processors.
        // Miri emulates a "hardware" performance counter with a resolution of 1ns,
        // and thus 10^9 counts per second.
        ecx.write_scalar(Scalar::from_i64(1_000_000_000), &ecx.deref_pointer(lpFrequency_op)?)?;
        interp_ok(Scalar::from_i32(-1)) // Return non-zero on success
    }

    #[allow(clippy::arithmetic_side_effects)]
    fn system_time_since_windows_epoch(&self, time: &SystemTime) -> InterpResult<'tcx, Duration> {
        // The amount of seconds between 1601/1/1 and 1970/1/1.
        // See https://learn.microsoft.com/en-us/windows/win32/sysinfo/converting-a-time-t-value-to-a-file-time
        // (just divide by the number of 100 ns intervals per second).
        const SECONDS_TO_UNIX_EPOCH: u64 = 11_644_473_600;

        interp_ok(system_time_to_duration(time)? + Duration::from_secs(SECONDS_TO_UNIX_EPOCH))
    }

    #[allow(non_snake_case, clippy::arithmetic_side_effects)]
    fn windows_ticks_for(&self, duration: Duration) -> InterpResult<'tcx, u64> {
        // 1 interval = 100 ns.
        // See https://learn.microsoft.com/en-us/windows/win32/api/minwinbase/ns-minwinbase-filetime
        const NANOS_PER_INTERVAL: u128 = 100;

        let ticks = u64::try_from(duration.as_nanos() / NANOS_PER_INTERVAL)
            .map_err(|_| err_unsup_format!("programs running more than 2^64 Windows ticks after the Windows epoch are not supported"))?;
        interp_ok(ticks)
    }

    fn mach_absolute_time(&self) -> InterpResult<'tcx, Scalar> {
        let ecx = self.eval_context_ref();

        ecx.assert_target_os(Os::MacOs, "mach_absolute_time");

        // This returns a u64, with time units determined dynamically by `mach_timebase_info`.
        // We return plain nanoseconds.
        let duration =
            ecx.machine.monotonic_clock.now().duration_since(ecx.machine.monotonic_clock.epoch());
        let res = u64::try_from(duration.as_nanos()).map_err(|_| {
            err_unsup_format!("programs running longer than 2^64 nanoseconds are not supported")
        })?;
        interp_ok(Scalar::from_u64(res))
    }

    fn mach_timebase_info(&mut self, info_op: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let ecx = self.eval_context_mut();

        ecx.assert_target_os(Os::MacOs, "mach_timebase_info");

        let info = ecx.deref_pointer(info_op)?;

        // Since our emulated ticks in `mach_absolute_time` *are* nanoseconds,
        // no scaling needs to happen.
        let (numerator, denom) = (1, 1);
        ecx.write_int_fields(&[numerator.into(), denom.into()], &info)?;

        interp_ok(Scalar::from_i32(0)) // KERN_SUCCESS
    }

    fn mach_wait_until(&mut self, deadline_op: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let ecx = self.eval_context_mut();

        ecx.assert_target_os(Os::MacOs, "mach_wait_until");

        let deadline = ecx.read_scalar(deadline_op)?.to_u64()?;
        // Our mach_absolute_time "ticks" are plain nanoseconds.
        let deadline = Duration::from_nanos(deadline);
        // This is *absolute* time.
        let deadline = ecx.machine.monotonic_clock.epoch().add_lossy(deadline);

        ecx.block_thread(
            BlockReason::Sleep,
            Some(deadline.into()),
            callback!(
                @capture<'tcx> {}
                |_ecx, unblock: UnblockKind| {
                    assert_eq!(unblock, UnblockKind::TimedOut);
                    interp_ok(())
                }
            ),
        );

        interp_ok(Scalar::from_i32(0)) // KERN_SUCCESS
    }

    fn nanosleep(&mut self, duration: &OpTy<'tcx>, rem: &OpTy<'tcx>) -> InterpResult<'tcx, Scalar> {
        let ecx = self.eval_context_mut();

        ecx.assert_target_os_is_unix("nanosleep");

        let duration = ecx.deref_pointer(duration)?;
        let _rem = ecx.read_pointer(rem)?; // Signal handlers are not supported, so rem will never be written to.

        let Some(duration) = ecx.read_timespec(&duration)? else {
            return ecx.set_errno_and_return_neg1_i32(LibcError("EINVAL"));
        };
        let deadline = ecx.machine.monotonic_clock.now().add_lossy(duration);

        ecx.block_thread(
            BlockReason::Sleep,
            Some(deadline.into()),
            callback!(
                @capture<'tcx> {}
                |_ecx, unblock: UnblockKind| {
                    assert_eq!(unblock, UnblockKind::TimedOut);
                    interp_ok(())
                }
            ),
        );
        interp_ok(Scalar::from_i32(0))
    }

    fn clock_nanosleep(
        &mut self,
        clock_id: &OpTy<'tcx>,
        flags: &OpTy<'tcx>,
        timespec: &OpTy<'tcx>,
        rem: &OpTy<'tcx>,
    ) -> InterpResult<'tcx, Scalar> {
        let ecx = self.eval_context_mut();

        let clock_id = ecx.read_scalar(clock_id)?.to_i32()?;
        let timespec = ecx.deref_pointer(timespec)?;
        let flags = ecx.read_scalar(flags)?.to_i32()?;
        let _rem = ecx.read_pointer(rem)?; // Signal handlers are not supported, so rem will never be written to.

        // The standard lib through sleep_until only needs CLOCK_MONOTONIC
        if clock_id != ecx.eval_libc_i32("CLOCK_MONOTONIC") {
            throw_unsup_format!("clock_nanosleep: only CLOCK_MONOTONIC is supported");
        }

        let Some(duration) = ecx.read_timespec(&timespec)? else {
            return ecx.set_errno_and_return_neg1_i32(LibcError("EINVAL"));
        };

        let timeout_style = if flags == 0 {
            // No flags set, the timespec should be interpreted as a duration
            // to sleep for, i.e., a relative time.
            TimeoutStyle::Relative
        } else if flags == ecx.eval_libc_i32("TIMER_ABSTIME") {
            // Only flag TIMER_ABSTIME set, the timespec should be interpreted as
            // an absolute time.
            TimeoutStyle::Absolute
        } else {
            // The standard lib (through `sleep_until`) only needs TIMER_ABSTIME
            throw_unsup_format!(
                "`clock_nanosleep` unsupported flags {flags}, only no flags or \
                TIMER_ABSTIME is supported"
            );
        };
        let deadline = ecx.machine.timeout(TimeoutClock::Monotonic, timeout_style, duration);

        ecx.block_thread(
            BlockReason::Sleep,
            Some(deadline),
            callback!(
                @capture<'tcx> {}
                |_ecx, unblock: UnblockKind| {
                    assert_eq!(unblock, UnblockKind::TimedOut);
                    interp_ok(())
                }
            ),
        );
        interp_ok(Scalar::from_i32(0))
    }

    #[allow(non_snake_case)]
    fn Sleep(&mut self, timeout: &OpTy<'tcx>) -> InterpResult<'tcx> {
        let ecx = self.eval_context_mut();

        ecx.assert_target_os(Os::Windows, "Sleep");

        let timeout_ms = ecx.read_scalar(timeout)?.to_u32()?;

        let duration = Duration::from_millis(timeout_ms.into());
        let deadline = ecx.machine.monotonic_clock.now().add_lossy(duration);

        ecx.block_thread(
            BlockReason::Sleep,
            Some(deadline.into()),
            callback!(
                @capture<'tcx> {}
                |_ecx, unblock: UnblockKind| {
                    assert_eq!(unblock, UnblockKind::TimedOut);
                    interp_ok(())
                }
            ),
        );
        interp_ok(())
    }

    /// Parse a `timespec` struct and return it as a [`Duration`]. It returns [`None`]
    /// if the value in the `timespec` struct is invalid. Some libc functions will return
    /// EINVAL in this case.
    fn read_timespec(&self, tp: &MPlaceTy<'tcx>) -> InterpResult<'tcx, Option<Duration>> {
        let ecx = self.eval_context_ref();

        let sec_field = ecx.project_field_named(tp, "tv_sec")?;
        let sec = ecx.read_int::<i128>(&sec_field)?;
        let nsec_field = ecx.project_field_named(tp, "tv_nsec")?;
        let nsec = ecx.read_int::<i128>(&nsec_field)?;

        // tv_sec can be negative; this does not actually return EINVAL natively.
        // Since we cannot represent a negative duration, we just don't support this for now.
        if sec < 0 {
            throw_unsup_format!("timespec with negative tv_sec is not supported");
        }
        interp_ok(try {
            let seconds: u64 = sec.try_into().expect("tv_sec should fit in 64 bits");
            // tv_nsec must be non-negative.
            let nanoseconds: u32 = nsec.try_into().ok()?;
            if nanoseconds >= 1_000_000_000 {
                // tv_nsec must not be greater than 999,999,999.
                None?
            }
            Duration::new(seconds, nanoseconds)
        })
    }

    /// Parse a `timeval` struct and return it as a [`Duration`]. It returns [`None`]
    /// if the value in the `timeval` struct is invalid. Some libc functions will return
    /// EINVAL in this case.
    fn read_timeval(&self, tp: &MPlaceTy<'tcx>) -> InterpResult<'tcx, Option<Duration>> {
        let ecx = self.eval_context_ref();

        let sec_field = ecx.project_field_named(tp, "tv_sec")?;
        let sec = ecx.read_int::<i128>(&sec_field)?;
        let usec_field = ecx.project_field_named(tp, "tv_usec")?;
        let usec = ecx.read_int::<i128>(&usec_field)?;

        // tv_sec can be negative; this does not actually return EINVAL natively.
        // Since we cannot represent a negative duration, we just don't support this for now.
        if sec < 0 {
            throw_unsup_format!("timespec with negative tv_sec is not supported");
        }
        interp_ok(try {
            let seconds: u64 = sec.try_into().expect("tv_sec should fit in 64 bits");
            // tv_usec must be non-negative.
            let microseconds: u32 = usec.try_into().ok()?;
            if microseconds >= 1_000_000 {
                // tv_usec must not be greater than 999,999.
                None?
            }
            Duration::new(seconds, microseconds.strict_mul(1000))
        })
    }
}
