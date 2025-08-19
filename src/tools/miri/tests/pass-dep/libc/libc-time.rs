//@ignore-target: windows # no libc time APIs on Windows
//@compile-flags: -Zmiri-disable-isolation
use std::time::{Duration, Instant};
use std::{env, mem, ptr};

fn main() {
    test_clocks();
    test_posix_gettimeofday();
    test_localtime_r_gmt();
    test_localtime_r_pst();
    test_localtime_r_epoch();
    #[cfg(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "freebsd",
        target_os = "android"
    ))]
    test_localtime_r_multiple_calls_deduplication();
    // Architecture-specific tests.
    #[cfg(target_pointer_width = "32")]
    test_localtime_r_future_32b();
    #[cfg(target_pointer_width = "64")]
    test_localtime_r_future_64b();

    test_nanosleep();
    #[cfg(any(
        target_os = "freebsd",
        target_os = "linux",
        target_os = "android",
        target_os = "solaris",
        target_os = "illumos"
    ))]
    {
        test_clock_nanosleep::absolute();
        test_clock_nanosleep::relative();
    }
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

/// Helper function to create an empty tm struct.
fn create_empty_tm() -> libc::tm {
    libc::tm {
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
    }
}

/// Original GMT test
fn test_localtime_r_gmt() {
    // Set timezone to GMT.
    let key = "TZ";
    env::set_var(key, "GMT");
    const TIME_SINCE_EPOCH: libc::time_t = 1712475836; // 2024-04-07 07:43:56 GMT
    let custom_time_ptr = &TIME_SINCE_EPOCH;
    let mut tm = create_empty_tm();
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
    {
        assert_eq!(tm.tm_gmtoff, 0);
        unsafe {
            assert_eq!(std::ffi::CStr::from_ptr(tm.tm_zone).to_str().unwrap(), "+00");
        }
    }

    // The returned value is the pointer passed in.
    assert!(ptr::eq(res, &mut tm));

    // Remove timezone setting.
    env::remove_var(key);
}

/// PST timezone test (testing different timezone handling).
fn test_localtime_r_pst() {
    let key = "TZ";
    env::set_var(key, "PST8PDT");
    const TIME_SINCE_EPOCH: libc::time_t = 1712475836; // 2024-04-07 07:43:56 GMT
    let custom_time_ptr = &TIME_SINCE_EPOCH;
    let mut tm = create_empty_tm();

    let res = unsafe { libc::localtime_r(custom_time_ptr, &mut tm) };

    assert_eq!(tm.tm_sec, 56);
    assert_eq!(tm.tm_min, 43);
    assert_eq!(tm.tm_hour, 0); // 7 - 7 = 0 (PDT offset)
    assert_eq!(tm.tm_mday, 7);
    assert_eq!(tm.tm_mon, 3);
    assert_eq!(tm.tm_year, 124);
    assert_eq!(tm.tm_wday, 0);
    assert_eq!(tm.tm_yday, 97);
    assert_eq!(tm.tm_isdst, -1); // DST information unavailable

    #[cfg(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "freebsd",
        target_os = "android"
    ))]
    {
        assert_eq!(tm.tm_gmtoff, -7 * 3600); // -7 hours in seconds
        unsafe {
            assert_eq!(std::ffi::CStr::from_ptr(tm.tm_zone).to_str().unwrap(), "-07");
        }
    }

    assert!(ptr::eq(res, &mut tm));
    env::remove_var(key);
}

/// Unix epoch test (edge case testing).
fn test_localtime_r_epoch() {
    let key = "TZ";
    env::set_var(key, "GMT");
    const TIME_SINCE_EPOCH: libc::time_t = 0; // 1970-01-01 00:00:00
    let custom_time_ptr = &TIME_SINCE_EPOCH;
    let mut tm = create_empty_tm();

    let res = unsafe { libc::localtime_r(custom_time_ptr, &mut tm) };

    assert_eq!(tm.tm_sec, 0);
    assert_eq!(tm.tm_min, 0);
    assert_eq!(tm.tm_hour, 0);
    assert_eq!(tm.tm_mday, 1);
    assert_eq!(tm.tm_mon, 0);
    assert_eq!(tm.tm_year, 70);
    assert_eq!(tm.tm_wday, 4); // Thursday
    assert_eq!(tm.tm_yday, 0);
    assert_eq!(tm.tm_isdst, -1);

    #[cfg(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "freebsd",
        target_os = "android"
    ))]
    {
        assert_eq!(tm.tm_gmtoff, 0);
        unsafe {
            assert_eq!(std::ffi::CStr::from_ptr(tm.tm_zone).to_str().unwrap(), "+00");
        }
    }

    assert!(ptr::eq(res, &mut tm));
    env::remove_var(key);
}

/// Future date test (testing large values).
#[cfg(target_pointer_width = "64")]
fn test_localtime_r_future_64b() {
    let key = "TZ";
    env::set_var(key, "GMT");

    // Using 2050-01-01 00:00:00 for 64-bit systems
    // value that's safe for 64-bit time_t
    const TIME_SINCE_EPOCH: libc::time_t = 2524608000;
    let custom_time_ptr = &TIME_SINCE_EPOCH;
    let mut tm = create_empty_tm();

    let res = unsafe { libc::localtime_r(custom_time_ptr, &mut tm) };

    assert_eq!(tm.tm_sec, 0);
    assert_eq!(tm.tm_min, 0);
    assert_eq!(tm.tm_hour, 0);
    assert_eq!(tm.tm_mday, 1);
    assert_eq!(tm.tm_mon, 0);
    assert_eq!(tm.tm_year, 150); // 2050 - 1900
    assert_eq!(tm.tm_wday, 6); // Saturday
    assert_eq!(tm.tm_yday, 0);
    assert_eq!(tm.tm_isdst, -1);

    #[cfg(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "freebsd",
        target_os = "android"
    ))]
    {
        assert_eq!(tm.tm_gmtoff, 0);
        unsafe {
            assert_eq!(std::ffi::CStr::from_ptr(tm.tm_zone).to_str().unwrap(), "+00");
        }
    }

    assert!(ptr::eq(res, &mut tm));
    env::remove_var(key);
}

/// Future date test (testing large values for 32b target).
#[cfg(target_pointer_width = "32")]
fn test_localtime_r_future_32b() {
    let key = "TZ";
    env::set_var(key, "GMT");

    // Using 2030-01-01 00:00:00 for 32-bit systems
    // Safe value within i32 range
    const TIME_SINCE_EPOCH: libc::time_t = 1893456000;
    let custom_time_ptr = &TIME_SINCE_EPOCH;
    let mut tm = create_empty_tm();

    let res = unsafe { libc::localtime_r(custom_time_ptr, &mut tm) };

    // Verify 2030-01-01 00:00:00
    assert_eq!(tm.tm_sec, 0);
    assert_eq!(tm.tm_min, 0);
    assert_eq!(tm.tm_hour, 0);
    assert_eq!(tm.tm_mday, 1);
    assert_eq!(tm.tm_mon, 0);
    assert_eq!(tm.tm_year, 130); // 2030 - 1900
    assert_eq!(tm.tm_wday, 2); // Tuesday
    assert_eq!(tm.tm_yday, 0);
    assert_eq!(tm.tm_isdst, -1);

    #[cfg(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "freebsd",
        target_os = "android"
    ))]
    {
        assert_eq!(tm.tm_gmtoff, 0);
        unsafe {
            assert_eq!(std::ffi::CStr::from_ptr(tm.tm_zone).to_str().unwrap(), "+00");
        }
    }

    assert!(ptr::eq(res, &mut tm));
    env::remove_var(key);
}

/// Tests the behavior of `localtime_r` with multiple calls to ensure deduplication of `tm_zone` pointers.
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "freebsd", target_os = "android"))]
fn test_localtime_r_multiple_calls_deduplication() {
    let key = "TZ";
    env::set_var(key, "PST8PDT");

    const TIME_SINCE_EPOCH_BASE: libc::time_t = 1712475836; // Base timestamp: 2024-04-07 07:43:56 GMT
    const NUM_CALLS: usize = 50;

    let mut unique_pointers = std::collections::HashSet::new();

    for i in 0..NUM_CALLS {
        let timestamp = TIME_SINCE_EPOCH_BASE + (i as libc::time_t * 3600); // Increment by 1 hour for each call
        let mut tm: libc::tm = create_empty_tm();
        let tm_ptr = unsafe { libc::localtime_r(&timestamp, &mut tm) };

        assert!(!tm_ptr.is_null(), "localtime_r failed for timestamp {timestamp}");

        unique_pointers.insert(tm.tm_zone);
    }

    let unique_count = unique_pointers.len();

    assert!(
        unique_count >= 2 && unique_count <= (NUM_CALLS - 1),
        "Unexpected number of unique tm_zone pointers: {} (expected between 2 and {})",
        unique_count,
        NUM_CALLS - 1
    );
}

fn test_nanosleep() {
    let start_test_sleep = Instant::now();
    let duration_zero = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    let remainder = ptr::null_mut::<libc::timespec>();
    let is_error = unsafe { libc::nanosleep(&duration_zero, remainder) };
    assert_eq!(is_error, 0);
    assert!(start_test_sleep.elapsed() < Duration::from_millis(100));

    let start_test_sleep = Instant::now();
    let duration_100_millis = libc::timespec { tv_sec: 0, tv_nsec: 1_000_000_000 / 10 };
    let remainder = ptr::null_mut::<libc::timespec>();
    let is_error = unsafe { libc::nanosleep(&duration_100_millis, remainder) };
    assert_eq!(is_error, 0);
    assert!(start_test_sleep.elapsed() > Duration::from_millis(100));
}

#[cfg(any(
    target_os = "freebsd",
    target_os = "linux",
    target_os = "android",
    target_os = "solaris",
    target_os = "illumos"
))]
mod test_clock_nanosleep {
    use super::*;

    /// Helper function used to create an instant in the future
    fn add_100_millis(mut ts: libc::timespec) -> libc::timespec {
        // While tv_nsec has type `c_long` tv_sec has type `time_t`. These might
        // end up as different types (for example: like i32 and i64).
        const SECOND: libc::c_long = 1_000_000_000;
        ts.tv_nsec += SECOND / 10;
        // If this pushes tv_nsec to SECOND or higher, we need to overflow to tv_sec.
        ts.tv_sec += (ts.tv_nsec / SECOND) as libc::time_t;
        ts.tv_nsec %= SECOND;
        ts
    }

    /// Helper function to get the current time for testing relative sleeps
    fn timespec_now(clock: libc::clockid_t) -> libc::timespec {
        let mut timespec = mem::MaybeUninit::<libc::timespec>::uninit();
        let is_error = unsafe { libc::clock_gettime(clock, timespec.as_mut_ptr()) };
        assert_eq!(is_error, 0);
        unsafe { timespec.assume_init() }
    }

    pub fn absolute() {
        let start_test_sleep = Instant::now();
        let before_start = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        let remainder = ptr::null_mut::<libc::timespec>();
        let error = unsafe {
            // this will not sleep since unix time zero is in the past
            libc::clock_nanosleep(
                libc::CLOCK_MONOTONIC,
                libc::TIMER_ABSTIME,
                &before_start,
                remainder,
            )
        };
        assert_eq!(error, 0);
        assert!(start_test_sleep.elapsed() < Duration::from_millis(100));

        let start_test_sleep = Instant::now();
        let hunderd_millis_after_start = add_100_millis(timespec_now(libc::CLOCK_MONOTONIC));
        let remainder = ptr::null_mut::<libc::timespec>();
        let error = unsafe {
            libc::clock_nanosleep(
                libc::CLOCK_MONOTONIC,
                libc::TIMER_ABSTIME,
                &hunderd_millis_after_start,
                remainder,
            )
        };
        assert_eq!(error, 0);
        assert!(start_test_sleep.elapsed() > Duration::from_millis(100));
    }

    pub fn relative() {
        const NO_FLAGS: i32 = 0;

        let start_test_sleep = Instant::now();
        let duration_zero = libc::timespec { tv_sec: 0, tv_nsec: 0 };
        let remainder = ptr::null_mut::<libc::timespec>();
        let error = unsafe {
            libc::clock_nanosleep(libc::CLOCK_MONOTONIC, NO_FLAGS, &duration_zero, remainder)
        };
        assert_eq!(error, 0);
        assert!(start_test_sleep.elapsed() < Duration::from_millis(100));

        let start_test_sleep = Instant::now();
        let duration_100_millis = libc::timespec { tv_sec: 0, tv_nsec: 1_000_000_000 / 10 };
        let remainder = ptr::null_mut::<libc::timespec>();
        let error = unsafe {
            libc::clock_nanosleep(libc::CLOCK_MONOTONIC, NO_FLAGS, &duration_100_millis, remainder)
        };
        assert_eq!(error, 0);
        assert!(start_test_sleep.elapsed() > Duration::from_millis(100));
    }
}
