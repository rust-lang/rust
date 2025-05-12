//@only-target: darwin
//@compile-flags: -Zmiri-deterministic-concurrency

use std::time::{Duration, Instant};
use std::{io, ptr, thread};

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

    // Wait for 200ms, with nobody waking us up early.
    unsafe {
        assert_eq!(
            libc::os_sync_wait_on_address_with_timeout(
                ptr::from_ref(&futex).cast_mut().cast(),
                123,
                size_of::<i32>(),
                libc::OS_SYNC_WAIT_ON_ADDRESS_NONE,
                libc::OS_CLOCK_MACH_ABSOLUTE_TIME,
                200_000_000,
            ),
            -1,
        );
        assert_eq!(io::Error::last_os_error().raw_os_error().unwrap(), libc::ETIMEDOUT);
    }

    assert!((200..1000).contains(&start.elapsed().as_millis()));
}

fn wait_absolute_timeout() {
    let start = Instant::now();

    // Get the current monotonic timestamp.
    #[allow(deprecated)]
    let mut deadline = unsafe { libc::mach_absolute_time() };

    // Add 200ms.
    // What we should be doing here is call `mach_timebase_info` to determine the
    // unit used for `deadline`, but we know what Miri returns for that function:
    // the unit is nanoseconds.
    deadline += 200_000_000;

    let futex: i32 = 123;

    // Wait for 200ms from now, with nobody waking us up early.
    unsafe {
        assert_eq!(
            libc::os_sync_wait_on_address_with_deadline(
                ptr::from_ref(&futex).cast_mut().cast(),
                123,
                size_of::<i32>(),
                libc::OS_SYNC_WAIT_ON_ADDRESS_NONE,
                libc::OS_CLOCK_MACH_ABSOLUTE_TIME,
                deadline,
            ),
            -1,
        );
        assert_eq!(io::Error::last_os_error().raw_os_error().unwrap(), libc::ETIMEDOUT);
    }

    assert!((200..1000).contains(&start.elapsed().as_millis()));
}

fn wait_wake() {
    let start = Instant::now();

    static mut FUTEX: i32 = 0;

    let t = thread::spawn(move || {
        thread::sleep(Duration::from_millis(200));
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

    // When running this in stress-gc mode, things can take quite long.
    // So the timeout is 3000 ms.
    assert!((200..3000).contains(&start.elapsed().as_millis()));
    t.join().unwrap();
}

fn wait_wake_multiple() {
    let val = 0i32;
    let futex = &val;

    thread::scope(|s| {
        // Spawn some threads and make them wait on the futex.
        for i in 0..4 {
            s.spawn(move || unsafe {
                assert_eq!(
                    libc::os_sync_wait_on_address(
                        ptr::from_ref(futex).cast_mut().cast(),
                        0,
                        size_of::<i32>(),
                        libc::OS_SYNC_WAIT_ON_ADDRESS_NONE,
                    ),
                    // The last two threads will be woken at the same time,
                    // but for the first two threads the remaining number
                    // of waiters should be strictly decreasing.
                    if i < 2 { 3 - i } else { 0 },
                );
            });

            thread::yield_now();
        }

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

            assert_eq!(
                libc::os_sync_wake_by_address_any(
                    ptr::from_ref(futex).cast_mut().cast(),
                    size_of::<i32>(),
                    libc::OS_SYNC_WAKE_BY_ADDRESS_NONE,
                ),
                0
            );

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
    })
}

fn param_mismatch() {
    let futex = 0;
    thread::scope(|s| {
        s.spawn(|| {
            unsafe {
                assert_eq!(
                    libc::os_sync_wait_on_address_with_timeout(
                        ptr::from_ref(&futex).cast_mut().cast(),
                        0,
                        size_of::<i32>(),
                        libc::OS_SYNC_WAIT_ON_ADDRESS_NONE,
                        libc::OS_CLOCK_MACH_ABSOLUTE_TIME,
                        400_000_000,
                    ),
                    -1,
                );
                assert_eq!(io::Error::last_os_error().raw_os_error().unwrap(), libc::ETIMEDOUT);
            }
        });

        s.spawn(|| {
            thread::yield_now();
            unsafe {
                assert_eq!(
                    libc::os_sync_wait_on_address(
                        ptr::from_ref(&futex).cast_mut().cast(),
                        0,
                        size_of::<i32>(),
                        libc::OS_SYNC_WAIT_ON_ADDRESS_SHARED,
                    ),
                    -1,
                );
                // This call fails because it uses the shared flag whereas the first waiter didn't.
                assert_eq!(io::Error::last_os_error().raw_os_error().unwrap(), libc::EINVAL);
            }
        });

        thread::yield_now();

        unsafe {
            assert_eq!(
                libc::os_sync_wake_by_address_any(
                    ptr::from_ref(&futex).cast_mut().cast(),
                    size_of::<i32>(),
                    libc::OS_SYNC_WAIT_ON_ADDRESS_SHARED,
                ),
                -1,
            );
            // This call fails because it uses the shared flag whereas the waiter didn't.
            assert_eq!(io::Error::last_os_error().raw_os_error().unwrap(), libc::EINVAL);
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
