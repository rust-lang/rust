// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Temporal quantification.

use libc;

pub use self::duration::Duration;

pub mod duration;

/// Returns the current value of a high-resolution performance counter
/// in nanoseconds since an unspecified epoch.
// NB: this is intentionally not public, this is not ready to stabilize its api.
fn precise_time_ns() -> u64 {
    return os_precise_time_ns();

    #[cfg(windows)]
    fn os_precise_time_ns() -> u64 {
        let mut ticks_per_s = 0;
        assert_eq!(unsafe {
            libc::QueryPerformanceFrequency(&mut ticks_per_s)
        }, 1);
        let ticks_per_s = if ticks_per_s == 0 {1} else {ticks_per_s};
        let mut ticks = 0;
        assert_eq!(unsafe {
            libc::QueryPerformanceCounter(&mut ticks)
        }, 1);

        return (ticks as u64 * 1000000000) / (ticks_per_s as u64);
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn os_precise_time_ns() -> u64 {
        use sync;

        static mut TIMEBASE: libc::mach_timebase_info = libc::mach_timebase_info { numer: 0,
                                                                                   denom: 0 };
        static ONCE: sync::Once = sync::ONCE_INIT;
        unsafe {
            ONCE.call_once(|| {
                imp::mach_timebase_info(&mut TIMEBASE);
            });
            let time = imp::mach_absolute_time();
            time * TIMEBASE.numer as u64 / TIMEBASE.denom as u64
        }
    }

    #[cfg(not(any(windows, target_os = "macos", target_os = "ios")))]
    fn os_precise_time_ns() -> u64 {
        let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        unsafe {
            imp::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts);
        }
        return (ts.tv_sec as u64) * 1000000000 + (ts.tv_nsec as u64)
    }
}

#[cfg(all(unix, not(target_os = "macos"), not(target_os = "ios")))]
mod imp {
    use libc::{c_int, timespec};

    // Apparently android provides this in some other library?
    #[cfg(not(target_os = "android"))]
    #[link(name = "rt")]
    extern {}

    extern {
        pub fn clock_gettime(clk_id: c_int, tp: *mut timespec) -> c_int;
    }

}
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod imp {
    use libc::{c_int, mach_timebase_info};

    extern {
        pub fn mach_absolute_time() -> u64;
        pub fn mach_timebase_info(info: *mut mach_timebase_info) -> c_int;
    }
}
