//@only-target: freebsd
//@compile-flags: -Zmiri-preemption-rate=0 -Zmiri-disable-isolation

use std::mem::{self, MaybeUninit};
use std::ptr::{self, addr_of};
use std::sync::atomic::AtomicU32;
use std::time::Instant;
use std::{io, thread};

fn wait_wake() {
    fn wake_nobody() {
        // Current thread waits on futex.
        // New thread wakes up 0 threads waiting on that futex.
        // Current thread should time out.
        static mut FUTEX: u32 = 0;

        let waker = thread::spawn(|| {
            unsafe {
                assert_eq!(
                    libc::_umtx_op(
                        addr_of!(FUTEX) as *mut _,
                        libc::UMTX_OP_WAKE_PRIVATE,
                        0, // wake up 0 waiters
                        ptr::null_mut::<libc::c_void>(),
                        ptr::null_mut::<libc::c_void>(),
                    ),
                    0
                );
            }
        });

        // 10ms should be enough.
        let mut timeout = libc::timespec { tv_sec: 0, tv_nsec: 10_000_000 };
        let timeout_size_arg =
            ptr::without_provenance_mut::<libc::c_void>(mem::size_of::<libc::timespec>());
        unsafe {
            assert_eq!(
                libc::_umtx_op(
                    addr_of!(FUTEX) as *mut _,
                    libc::UMTX_OP_WAIT_UINT_PRIVATE,
                    0,
                    timeout_size_arg,
                    &mut timeout as *mut _ as _,
                ),
                -1
            );
            // Main thread did not get woken up, so it timed out.
            assert_eq!(io::Error::last_os_error().raw_os_error().unwrap(), libc::ETIMEDOUT);
        }

        waker.join().unwrap();
    }

    fn wake_two_of_three() {
        // We create 2 threads that wait on a futex with a 100ms timeout.
        // The main thread wakes up 2 threads waiting on this futex and after this
        // checks that only those threads woke up and the other one timed out.
        static mut FUTEX: u32 = 0;

        fn waiter() -> bool {
            let mut timeout = libc::timespec { tv_sec: 0, tv_nsec: 100_000_000 };
            let timeout_size_arg =
                ptr::without_provenance_mut::<libc::c_void>(mem::size_of::<libc::timespec>());
            unsafe {
                libc::_umtx_op(
                    addr_of!(FUTEX) as *mut _,
                    libc::UMTX_OP_WAIT_UINT_PRIVATE,
                    0, // FUTEX is 0
                    timeout_size_arg,
                    &mut timeout as *mut _ as _,
                );
                // Return true if this thread woke up.
                io::Error::last_os_error().raw_os_error().unwrap() != libc::ETIMEDOUT
            }
        }

        let t1 = thread::spawn(waiter);
        let t2 = thread::spawn(waiter);
        let t3 = thread::spawn(waiter);

        // Run all the waiters, so they can go to sleep.
        thread::yield_now();

        // Wake up 2 thread and make sure 1 is still waiting.
        unsafe {
            assert_eq!(
                libc::_umtx_op(
                    addr_of!(FUTEX) as *mut _,
                    libc::UMTX_OP_WAKE_PRIVATE,
                    2,
                    ptr::null_mut::<libc::c_void>(),
                    ptr::null_mut::<libc::c_void>(),
                ),
                0
            );
        }

        // Treat the booleans as numbers to simplify checking how many threads were woken up.
        let t1 = t1.join().unwrap() as usize;
        let t2 = t2.join().unwrap() as usize;
        let t3 = t3.join().unwrap() as usize;
        let woken_up_count = t1 + t2 + t3;
        assert!(woken_up_count == 2, "Expected 2 threads to wake up got: {woken_up_count}");
    }

    wake_nobody();
    wake_two_of_three();
}

fn wake_dangling() {
    let futex = Box::new(0);
    let ptr: *const u32 = &*futex;
    drop(futex);

    // Expect error since this is now "unmapped" memory.
    unsafe {
        assert_eq!(
            libc::_umtx_op(
                ptr as *const AtomicU32 as *mut _,
                libc::UMTX_OP_WAKE_PRIVATE,
                0,
                ptr::null_mut::<libc::c_void>(),
                ptr::null_mut::<libc::c_void>(),
            ),
            -1
        );
        assert_eq!(io::Error::last_os_error().raw_os_error().unwrap(), libc::EFAULT);
    }
}

fn wait_wrong_val() {
    let futex: u32 = 123;

    // Wait with a wrong value just returns 0
    unsafe {
        assert_eq!(
            libc::_umtx_op(
                ptr::from_ref(&futex).cast_mut().cast(),
                libc::UMTX_OP_WAIT_UINT_PRIVATE,
                456,
                ptr::null_mut::<libc::c_void>(),
                ptr::null_mut::<libc::c_void>(),
            ),
            0
        );
    }
}

fn wait_relative_timeout() {
    fn without_timespec() {
        let start = Instant::now();

        let futex: u32 = 123;

        let mut timeout = libc::timespec { tv_sec: 0, tv_nsec: 200_000_000 };
        let timeout_size_arg =
            ptr::without_provenance_mut::<libc::c_void>(mem::size_of::<libc::timespec>());
        // Wait for 200ms, with nobody waking us up early
        unsafe {
            assert_eq!(
                libc::_umtx_op(
                    ptr::from_ref(&futex).cast_mut().cast(),
                    libc::UMTX_OP_WAIT_UINT_PRIVATE,
                    123,
                    timeout_size_arg,
                    &mut timeout as *mut _ as _,
                ),
                -1
            );
            assert_eq!(io::Error::last_os_error().raw_os_error().unwrap(), libc::ETIMEDOUT);
        }

        assert!((200..1000).contains(&start.elapsed().as_millis()));
    }

    fn with_timespec() {
        let futex: u32 = 123;
        let mut timeout = libc::_umtx_time {
            _timeout: libc::timespec { tv_sec: 0, tv_nsec: 200_000_000 },
            _flags: 0,
            _clockid: libc::CLOCK_MONOTONIC as u32,
        };
        let timeout_size_arg =
            ptr::without_provenance_mut::<libc::c_void>(mem::size_of::<libc::_umtx_time>());

        let start = Instant::now();

        // Wait for 200ms, with nobody waking us up early
        unsafe {
            assert_eq!(
                libc::_umtx_op(
                    ptr::from_ref(&futex).cast_mut().cast(),
                    libc::UMTX_OP_WAIT_UINT_PRIVATE,
                    123,
                    timeout_size_arg,
                    &mut timeout as *mut _ as _,
                ),
                -1
            );
            assert_eq!(io::Error::last_os_error().raw_os_error().unwrap(), libc::ETIMEDOUT);
        }
        assert!((200..1000).contains(&start.elapsed().as_millis()));
    }

    without_timespec();
    with_timespec();
}

fn wait_absolute_timeout() {
    let start = Instant::now();

    // Get the current monotonic timestamp as timespec.
    let mut timeout = unsafe {
        let mut now: MaybeUninit<libc::timespec> = MaybeUninit::uninit();
        assert_eq!(libc::clock_gettime(libc::CLOCK_MONOTONIC, now.as_mut_ptr()), 0);
        now.assume_init()
    };

    // Add 200ms.
    timeout.tv_nsec += 200_000_000;
    if timeout.tv_nsec > 1_000_000_000 {
        timeout.tv_nsec -= 1_000_000_000;
        timeout.tv_sec += 1;
    }

    // Create umtx_timeout struct with that absolute timeout.
    let umtx_timeout = libc::_umtx_time {
        _timeout: timeout,
        _flags: libc::UMTX_ABSTIME,
        _clockid: libc::CLOCK_MONOTONIC as u32,
    };
    let umtx_timeout_ptr = &umtx_timeout as *const _;
    let umtx_timeout_size = ptr::without_provenance_mut(mem::size_of_val(&umtx_timeout));

    let futex: u32 = 123;

    // Wait for 200ms from now, with nobody waking us up early.
    unsafe {
        assert_eq!(
            libc::_umtx_op(
                ptr::from_ref(&futex).cast_mut().cast(),
                libc::UMTX_OP_WAIT_UINT_PRIVATE,
                123,
                umtx_timeout_size,
                umtx_timeout_ptr as *mut _,
            ),
            -1
        );
        assert_eq!(io::Error::last_os_error().raw_os_error().unwrap(), libc::ETIMEDOUT);
    }
    assert!((200..1000).contains(&start.elapsed().as_millis()));
}

fn main() {
    wait_wake();
    wake_dangling();
    wait_wrong_val();
    wait_relative_timeout();
    wait_absolute_timeout();
}
