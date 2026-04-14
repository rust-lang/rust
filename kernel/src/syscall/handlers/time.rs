//! Time and scheduling syscalls

use abi::errors::SysResult;
use abi::time::{ClockId, TimeSpec};

use crate::syscall::validate::{copyout, validate_user_range};

pub fn sys_yield() -> SysResult<usize> {
    unsafe {
        crate::sched::yield_now_current();
    }
    Ok(0)
}

pub fn sys_sleep_ns(ns: u64) -> SysResult<usize> {
    let ticks = crate::time::duration_to_sleep_ticks(ns);
    if crate::sched::take_pending_interrupt_current() {
        return Err(abi::errors::Errno::EINTR);
    }
    if ticks == 0 {
        unsafe {
            crate::sched::yield_now_current();
        }
    } else {
        crate::sched::sleep_ticks_current(ticks);
    }
    if crate::sched::take_pending_interrupt_current() {
        return Err(abi::errors::Errno::EINTR);
    }
    Ok(0)
}

pub fn sys_sleep_ms(ms: u64) -> SysResult<usize> {
    sys_sleep_ns(ms.saturating_mul(1_000_000))
}

pub fn sys_time_monotonic_ns() -> SysResult<usize> {
    Ok(crate::time::monotonic_now_ns() as usize)
}

pub fn sys_time_now(clock_id_raw: u32, out_ptr: usize) -> SysResult<usize> {
    let clock_id = ClockId::from_u32(clock_id_raw).ok_or(abi::errors::Errno::EINVAL)?;
    validate_user_range(out_ptr, core::mem::size_of::<TimeSpec>(), true)?;

    let now_ns = match clock_id {
        ClockId::Monotonic => crate::time::monotonic_now_ns(),
        ClockId::Realtime => crate::time::realtime_now_ns().ok_or(abi::errors::Errno::EAGAIN)?,
    };
    let spec = TimeSpec::from_nanos(now_ns);
    let bytes = unsafe {
        core::slice::from_raw_parts(
            &spec as *const TimeSpec as *const u8,
            core::mem::size_of::<TimeSpec>(),
        )
    };
    unsafe {
        copyout(out_ptr, bytes)?;
    }
    Ok(0)
}

pub fn sys_time_anchor(unix_secs: u64) -> SysResult<usize> {
    crate::time::anchor_system_clock(unix_secs, crate::time::monotonic_now_ns());
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sched::hooks::{SLEEP_TICKS_HOOK, TAKE_PENDING_INTERRUPT_HOOK};
    use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    static INTERRUPT_PENDING: AtomicBool = AtomicBool::new(false);
    static SLEPT_TICKS: AtomicU64 = AtomicU64::new(0);

    fn take_interrupt() -> bool {
        INTERRUPT_PENDING.swap(false, Ordering::SeqCst)
    }

    fn record_sleep_ticks(ticks: u64) {
        SLEPT_TICKS.store(ticks, Ordering::SeqCst);
    }

    #[test]
    fn sleep_returns_eintr_when_interrupt_is_already_pending() {
        unsafe {
            TAKE_PENDING_INTERRUPT_HOOK = Some(take_interrupt);
            SLEEP_TICKS_HOOK = Some(record_sleep_ticks);
        }
        INTERRUPT_PENDING.store(true, Ordering::SeqCst);
        SLEPT_TICKS.store(0, Ordering::SeqCst);

        let result = sys_sleep_ns(1_000_000);

        assert_eq!(result, Err(abi::errors::Errno::EINTR));
        assert_eq!(SLEPT_TICKS.load(Ordering::SeqCst), 0);
    }
}
