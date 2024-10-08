//@ignore-target: windows # no libc time APIs on Windows
//@compile-flags: -Zmiri-disable-isolation
use std::{env, mem, ptr};

fn main() {
    test_clocks();
    test_posix_gettimeofday();
    test_localtime_r();
}

/// Tests whether clock support exists at all
fn test_clocks() {
    let mut tp = mem::MaybeUninit::<libc::timespec>::uninit();
    let is_error = unsafe { libc::clock_gettime(libc::CLOCK_REALTIME, tp.as_mut_ptr()) };
    assert_eq!(is_error, 0);
    let is_error = unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, tp.as_mut_ptr()) };
    assert_eq!(is_error, 0);
    #[cfg(any(target_os = "linux", target_os = "freebsd", target_os = "android"))]
    {
        let is_error = unsafe { libc::clock_gettime(libc::CLOCK_REALTIME_COARSE, tp.as_mut_ptr()) };
        assert_eq!(is_error, 0);
        let is_error =
            unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC_COARSE, tp.as_mut_ptr()) };
        assert_eq!(is_error, 0);
    }
    #[cfg(target_os = "macos")]
    {
        let is_error = unsafe { libc::clock_gettime(libc::CLOCK_UPTIME_RAW, tp.as_mut_ptr()) };
        assert_eq!(is_error, 0);
    }
}

fn test_posix_gettimeofday() {
    let mut tp = mem::MaybeUninit::<libc::timeval>::uninit();
    let tz = ptr::null_mut::<libc::timezone>();
    let is_error = unsafe { libc::gettimeofday(tp.as_mut_ptr(), tz.cast()) };
    assert_eq!(is_error, 0);
    let tv = unsafe { tp.assume_init() };
    assert!(tv.tv_sec > 0);
    assert!(tv.tv_usec >= 0); // Theoretically this could be 0.

    // Test that non-null tz returns an error.
    let mut tz = mem::MaybeUninit::<libc::timezone>::uninit();
    let tz_ptr = tz.as_mut_ptr();
    let is_error = unsafe { libc::gettimeofday(tp.as_mut_ptr(), tz_ptr.cast()) };
    assert_eq!(is_error, -1);
}

fn test_localtime_r() {
    // Set timezone to GMT.
    let key = "TZ";
    env::set_var(key, "GMT");

    const TIME_SINCE_EPOCH: libc::time_t = 1712475836;
    let custom_time_ptr = &TIME_SINCE_EPOCH;
    let mut tm = libc::tm {
        tm_sec: 0,
        tm_min: 0,
        tm_hour: 0,
        tm_mday: 0,
        tm_mon: 0,
        tm_year: 0,
        tm_wday: 0,
        tm_yday: 0,
        tm_isdst: 0,
        #[cfg(any(
            target_os = "linux",
            target_os = "macos",
            target_os = "freebsd",
            target_os = "android"
        ))]
        tm_gmtoff: 0,
        #[cfg(any(
            target_os = "linux",
            target_os = "macos",
            target_os = "freebsd",
            target_os = "android"
        ))]
        tm_zone: std::ptr::null_mut::<libc::c_char>(),
    };
    let res = unsafe { libc::localtime_r(custom_time_ptr, &mut tm) };

    assert_eq!(tm.tm_sec, 56);
    assert_eq!(tm.tm_min, 43);
    assert_eq!(tm.tm_hour, 7);
    assert_eq!(tm.tm_mday, 7);
    assert_eq!(tm.tm_mon, 3);
    assert_eq!(tm.tm_year, 124);
    assert_eq!(tm.tm_wday, 0);
    assert_eq!(tm.tm_yday, 97);
    assert_eq!(tm.tm_isdst, -1);
    #[cfg(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "freebsd",
        target_os = "android"
    ))]
    assert_eq!(tm.tm_gmtoff, 0);
    #[cfg(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "freebsd",
        target_os = "android"
    ))]
    unsafe {
        assert_eq!(std::ffi::CStr::from_ptr(tm.tm_zone).to_str().unwrap(), "+00")
    };

    // The returned value is the pointer passed in.
    assert!(ptr::eq(res, &mut tm));

    // Remove timezone setting.
    env::remove_var(key);
}
