//@only-target: darwin
//@compile-flags: -Zmiri-fixed-schedule
//@run-native

use std::time::{Duration, Instant};
use std::{io, ptr, sync, thread};

#[path = "../../utils/libc.rs"]
mod libc_utils;
use libc_utils::errno_result;

fn yield_now() {
    if cfg!(miri) {
        // We control the scheduler so we know this is enough.
        thread::yield_now();
    } else {
        // Without scheduler control, we hope that sleeping a bit is enough.
        thread::sleep(Duration::from_millis(20));
    }
}

fn wake_nobody() {
    let futex = 0;

    // Wake 1 waiter. Expect ENOENT as nobody is waiting.
    unsafe {
        assert_eq!(
            libc::os_sync_wake_by_address_any(
                ptr::from_ref(&futex).cast_mut().cast(),
                size_of::<i32>(),
                libc::OS_SYNC_WAKE_BY_ADDRESS_NONE
            ),
            -1
        );
        assert_eq!(io::Error::last_os_error().raw_os_error().unwrap(), libc::ENOENT);
    }
}

fn wake_dangling() {
    let futex = Box::new(0);
    let ptr = ptr::from_ref(&futex).cast_mut().cast();
    drop(futex);

    // Expect error since this is now "unmapped" memory.
    unsafe {
        assert_eq!(
            libc::os_sync_wake_by_address_any(
                ptr,
                size_of::<i32>(),
                libc::OS_SYNC_WAKE_BY_ADDRESS_NONE
            ),
            -1
        );
        assert_eq!(io::Error::last_os_error().raw_os_error().unwrap(), libc::ENOENT);
    }
}

fn wait_wrong_val() {
    let futex: i32 = 123;

    // Only wait if the futex value is 456.
    unsafe {
        assert_eq!(
            libc::os_sync_wait_on_address(
                ptr::from_ref(&futex).cast_mut().cast(),
                456,
                size_of::<i32>(),
                libc::OS_SYNC_WAIT_ON_ADDRESS_NONE
            ),
            0,
        );
    }
}

fn wait_timeout() {
    let start = Instant::now();

    let futex: i32 = 123;

    // Wait for 100ms, with nobody waking us up early.
    unsafe {
        assert_eq!(
            libc::os_sync_wait_on_address_with_timeout(
                ptr::from_ref(&futex).cast_mut().cast(),
                123,
                size_of::<i32>(),
                libc::OS_SYNC_WAIT_ON_ADDRESS_NONE,
                libc::OS_CLOCK_MACH_ABSOLUTE_TIME,
                100_000_000,
            ),
            -1,
        );
        assert_eq!(io::Error::last_os_error().raw_os_error().unwrap(), libc::ETIMEDOUT);
    }

    assert!((100..1000).contains(&start.elapsed().as_millis()));
}

/// Compute a deadline that fires in `ms` milliseconds from now.
#[allow(deprecated)] // libc deprecated all the mach_time methods
fn deadline_in_ms(ms: u64) -> u64 {
    // Get clock resolution information.
    let mut timebase = libc::mach_timebase_info { numer: 0, denom: 0 };
    assert_eq!(unsafe { libc::mach_timebase_info(&mut timebase) }, libc::KERN_SUCCESS);

    // Convert delta from ms to mach time.
    let nanos = u128::from(ms) * 1_000_000;
    let delta =
        u64::try_from((nanos * u128::from(timebase.denom)).div_ceil(u128::from(timebase.numer)))
            .unwrap();

    (unsafe { libc::mach_absolute_time() }) + delta
}

fn wait_absolute_timeout() {
    let start = Instant::now();
    let futex: i32 = 123;

    // Wait for 100ms from now, with nobody waking us up early.
    unsafe {
        assert_eq!(
            libc::os_sync_wait_on_address_with_deadline(
                ptr::from_ref(&futex).cast_mut().cast(),
                123,
                size_of::<i32>(),
                libc::OS_SYNC_WAIT_ON_ADDRESS_NONE,
                libc::OS_CLOCK_MACH_ABSOLUTE_TIME,
                deadline_in_ms(100),
            ),
            -1,
        );
        assert_eq!(io::Error::last_os_error().raw_os_error().unwrap(), libc::ETIMEDOUT);
    }

    let time_passed = start.elapsed().as_millis();
    assert!((100..1000).contains(&time_passed), "slept for {time_passed}ms");
}

fn wait_wake() {
    static mut FUTEX: i32 = 0;

    let t = thread::spawn(move || {
        thread::sleep(Duration::from_millis(100));
        unsafe {
            assert_eq!(
                libc::os_sync_wake_by_address_any(
                    (&raw const FUTEX).cast_mut().cast(),
                    size_of::<i32>(),
                    libc::OS_SYNC_WAKE_BY_ADDRESS_NONE,
                ),
                0,
            );
        }
    });

    let start = Instant::now();

    unsafe {
        assert_eq!(
            libc::os_sync_wait_on_address(
                (&raw const FUTEX).cast_mut().cast(),
                0,
                size_of::<i32>(),
                libc::OS_SYNC_WAIT_ON_ADDRESS_NONE,
            ),
            0,
        );
    }

    let time_passed = start.elapsed().as_millis();
    assert!((100..1000).contains(&time_passed), "slept for {time_passed}ms");
    t.join().unwrap();
}

fn wait_wake_multiple() {
    let val = 0i32;
    let futex = &val;
    let remaining_waiters = sync::Mutex::new(Vec::new());

    thread::scope(|s| {
        // Spawn some threads and make them wait on the futex.
        for _ in 0..4 {
            s.spawn(|| unsafe {
                let res = errno_result(libc::os_sync_wait_on_address(
                    ptr::from_ref(futex).cast_mut().cast(),
                    0,
                    size_of::<i32>(),
                    libc::OS_SYNC_WAIT_ON_ADDRESS_NONE,
                ))
                .unwrap();
                remaining_waiters.lock().unwrap().push(res);
            });
        }
        // Let all threads get blocked.
        yield_now();

        // Wake the threads up again.
        unsafe {
            assert_eq!(
                libc::os_sync_wake_by_address_any(
                    ptr::from_ref(futex).cast_mut().cast(),
                    size_of::<i32>(),
                    libc::OS_SYNC_WAKE_BY_ADDRESS_NONE,
                ),
                0
            );

            yield_now();

            assert_eq!(
                libc::os_sync_wake_by_address_any(
                    ptr::from_ref(futex).cast_mut().cast(),
                    size_of::<i32>(),
                    libc::OS_SYNC_WAKE_BY_ADDRESS_NONE,
                ),
                0
            );

            yield_now();

            // Wake both remaining threads at the same time.
            assert_eq!(
                libc::os_sync_wake_by_address_all(
                    ptr::from_ref(futex).cast_mut().cast(),
                    size_of::<i32>(),
                    libc::OS_SYNC_WAKE_BY_ADDRESS_NONE,
                ),
                0
            );
        }
    });

    // Check `remaining_waiters` sequence. The first two must be "3, 2" as we wake them up
    // separately. On Miri the last two are woken together; natively they can also be one
    // after the other.
    let remaining_waiters = remaining_waiters.into_inner().unwrap();
    assert!(
        &remaining_waiters == &[3, 2, 0, 0]
            || (!cfg!(miri)
                && (&remaining_waiters == &[3, 2, 0, 1] || &remaining_waiters == &[3, 2, 1, 0])),
        "remaining_waiters sequence is {remaining_waiters:?}",
    )
}

fn param_mismatch() {
    let futex = 0;
    thread::scope(|s| {
        s.spawn(|| unsafe {
            let err = errno_result(libc::os_sync_wait_on_address_with_timeout(
                ptr::from_ref(&futex).cast_mut().cast(),
                0,
                size_of::<i32>(),
                libc::OS_SYNC_WAIT_ON_ADDRESS_NONE,
                libc::OS_CLOCK_MACH_ABSOLUTE_TIME,
                100_000_000,
            ))
            .unwrap_err();
            assert_eq!(err.raw_os_error().unwrap(), libc::ETIMEDOUT);
        });
        yield_now(); // let thread block

        unsafe {
            let err = errno_result(libc::os_sync_wake_by_address_any(
                ptr::from_ref(&futex).cast_mut().cast(),
                size_of::<i8>(),
                libc::OS_SYNC_WAIT_ON_ADDRESS_SHARED,
            ))
            .unwrap_err();
            // This call fails because there is a size mismatch
            assert_eq!(err.raw_os_error().unwrap(), libc::EINVAL);
        }
    });
}

fn main() {
    wake_nobody();
    wake_dangling();
    wait_wrong_val();
    wait_timeout();
    wait_absolute_timeout();
    wait_wake();
    wait_wake_multiple();
    param_mismatch();
}
