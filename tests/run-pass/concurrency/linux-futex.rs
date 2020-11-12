// Unfortunately, the test framework does not support 'only-linux',
// so we need to ignore Windows and macOS instead.
// ignore-macos: Uses Linux-only APIs
// ignore-windows: Uses Linux-only APIs
// compile-flags: -Zmiri-disable-isolation

#![feature(rustc_private)]
extern crate libc;

use std::ptr;
use std::thread;
use std::time::{Duration, Instant};

fn wake_nobody() {
    let futex = 0;

    // Wake 1 waiter. Expect zero waiters woken up, as nobody is waiting.
    unsafe {
        assert_eq!(libc::syscall(
            libc::SYS_futex,
            &futex as *const i32,
            libc::FUTEX_WAKE,
            1,
        ), 0);
    }

    // Same, but without omitting the unused arguments.
    unsafe {
        assert_eq!(libc::syscall(
            libc::SYS_futex,
            &futex as *const i32,
            libc::FUTEX_WAKE,
            1,
            0,
            0,
            0,
        ), 0);
    }
}

fn wake_dangling() {
    let futex = Box::new(0);
    let ptr: *const i32 = &*futex;
    drop(futex);

    // Wake 1 waiter. Expect zero waiters woken up, as nobody is waiting.
    unsafe {
        assert_eq!(libc::syscall(
            libc::SYS_futex,
            ptr,
            libc::FUTEX_WAKE,
            1,
        ), 0);
    }
}

fn wait_wrong_val() {
    let futex: i32 = 123;

    // Only wait if the futex value is 456.
    unsafe {
        assert_eq!(libc::syscall(
            libc::SYS_futex,
            &futex as *const i32,
            libc::FUTEX_WAIT,
            456,
            ptr::null::<libc::timespec>(),
        ), -1);
        assert_eq!(*libc::__errno_location(), libc::EAGAIN);
    }
}

fn wait_timeout() {
    let start = Instant::now();

    let futex: i32 = 123;

    // Wait for 200ms, with nobody waking us up early.
    unsafe {
        assert_eq!(libc::syscall(
            libc::SYS_futex,
            &futex as *const i32,
            libc::FUTEX_WAIT,
            123,
            &libc::timespec {
                tv_sec: 0,
                tv_nsec: 200_000_000,
            },
        ), -1);
        assert_eq!(*libc::__errno_location(), libc::ETIMEDOUT);
    }

    assert!((200..1000).contains(&start.elapsed().as_millis()));
}

fn wait_wake() {
    let start = Instant::now();

    static FUTEX: i32 = 0;

    thread::spawn(move || {
        thread::sleep(Duration::from_millis(200));
        unsafe {
            assert_eq!(libc::syscall(
                libc::SYS_futex,
                &FUTEX as *const i32,
                libc::FUTEX_WAKE,
                10, // Wake up at most 10 threads.
            ), 1); // Woken up one thread.
        }
    });

    unsafe {
        assert_eq!(libc::syscall(
            libc::SYS_futex,
            &FUTEX as *const i32,
            libc::FUTEX_WAIT,
            0,
            ptr::null::<libc::timespec>(),
        ), 0);
    }

    assert!((200..1000).contains(&start.elapsed().as_millis()));
}

fn main() {
    wake_nobody();
    wake_dangling();
    wait_wrong_val();
    wait_timeout();
    wait_wake();
}
