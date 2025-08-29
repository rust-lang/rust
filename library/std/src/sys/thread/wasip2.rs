use crate::ffi::CStr;
use crate::io;
use crate::num::NonZero;
use crate::time::{Duration, Instant};

pub struct Thread(!);

pub const DEFAULT_MIN_STACK_SIZE: usize = 1024 * 1024;

impl Thread {
    pub unsafe fn new(
        _stack: usize,
        _name: Option<&str>,
        _p: Box<dyn FnOnce()>,
    ) -> io::Result<Thread> {
        // Note that unlike WASIp1 even if the wasm `atomics` feature is enabled
        // there is no support for threads, not even experimentally, not even in
        // wasi-libc. Thus this is unconditionally unsupported.
        crate::sys::unsupported()
    }

    pub fn yield_now() {
        // no API for this in WASIp2, but there's also no threads, so that's
        // sort of expected.
    }

    pub fn set_name(_name: &CStr) {
        // nope
    }

    pub fn sleep(dur: Duration) {
        // Sleep in increments of `u64::MAX` nanoseconds until the `dur` is
        // entirely drained.
        let mut remaining = dur.as_nanos();
        while remaining > 0 {
            let amt = u64::try_from(remaining).unwrap_or(u64::MAX);
            wasip2::clocks::monotonic_clock::subscribe_duration(amt).block();
            remaining -= u128::from(amt);
        }
    }

    pub fn sleep_until(deadline: Instant) {
        match u64::try_from(deadline.into_inner().as_duration().as_nanos()) {
            // If the point in time we're sleeping to fits within a 64-bit
            // number of nanoseconds then directly use `subscribe_instant`.
            Ok(deadline) => {
                wasip2::clocks::monotonic_clock::subscribe_instant(deadline).block();
            }
            // ... otherwise we're sleeping for 500+ years relative to the
            // "start" of what the system is using as a clock so speed/accuracy
            // is not so much of a concern. Use `sleep` instead.
            Err(_) => {
                let now = Instant::now();

                if let Some(delay) = deadline.checked_duration_since(now) {
                    Self::sleep(delay);
                }
            }
        }
    }

    pub fn join(self) {
        self.0
    }
}

pub(crate) fn current_os_id() -> Option<u64> {
    None
}

pub fn available_parallelism() -> io::Result<NonZero<usize>> {
    crate::sys::unsupported()
}
