//@only-target: darwin
// We use `yield` to test specific interleavings, so disable automatic preemption.
//@compile-flags: -Zmiri-disable-isolation -Zmiri-deterministic-concurrency
#![allow(non_camel_case_types)]

use std::{ffi, thread, time};

type dispatch_semaphore_t = *mut ffi::c_void;
type dispatch_time_t = u64;

const DISPATCH_TIME_NOW: dispatch_time_t = 0;
const DISPATCH_TIME_FOREVER: dispatch_time_t = !0;

// Contained in libSystem.dylib, which is linked by default.
unsafe extern "C" {
    fn dispatch_time(when: dispatch_time_t, delta: i64) -> dispatch_time_t;
    fn dispatch_semaphore_create(val: isize) -> dispatch_semaphore_t;
    fn dispatch_semaphore_wait(dsema: dispatch_semaphore_t, timeout: dispatch_time_t) -> isize;
    fn dispatch_semaphore_signal(dsema: dispatch_semaphore_t) -> isize;
    fn dispatch_release(object: *mut ffi::c_void);
}

struct SendPtr<T> {
    ptr: *mut T,
}
unsafe impl<T> Send for SendPtr<T> {}
impl<T> Copy for SendPtr<T> {}
impl<T> Clone for SendPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

fn yield_now() {
    if cfg!(miri) {
        // We control the scheduler so we know this is enough.
        thread::yield_now();
    } else {
        // Without scheduler control, we hope that sleeping a bit is enough.
        thread::sleep(time::Duration::from_millis(10));
    }
}

fn main() {
    unsafe {
        let mut data = 0;
        let data = SendPtr { ptr: &mut data };

        let sem = SendPtr { ptr: dispatch_semaphore_create(1) };

        let res = dispatch_semaphore_wait(sem.ptr, DISPATCH_TIME_FOREVER);
        assert_eq!(res, 0);

        // Semaphore blocks when 0 is hit, and we can use that to synchronize.
        let t = thread::spawn(move || {
            let data = data;
            let sem = sem;

            let res = dispatch_semaphore_wait(sem.ptr, DISPATCH_TIME_FOREVER);
            assert_eq!(res, 0);
            let val = data.ptr.read();
            assert_eq!(val, 1);
        });
        yield_now(); // let `t` go first.
        // `t` will block so we end up here.
        data.ptr.write(1);
        let res = dispatch_semaphore_signal(sem.ptr);
        assert_eq!(res, 1); // non-zero value: a thread was woken up.
        t.join().unwrap();

        // Signaling again when there is nobody waiting returns 0.
        let res = dispatch_semaphore_signal(sem.ptr);
        assert_eq!(res, 0);

        // Setup a timeout.
        let before_time = time::Instant::now();
        let timeout = dispatch_time(DISPATCH_TIME_NOW, /* nanos */ 10 * 1000 * 1000);
        let res = dispatch_semaphore_wait(sem.ptr, timeout);
        assert_eq!(res, 0);
        let res = dispatch_semaphore_wait(sem.ptr, timeout);
        assert_eq!(res, 1);
        let elapsed_time = before_time.elapsed().as_millis();
        assert!(10 <= elapsed_time && elapsed_time <= 100);

        // Need to get the value back up to at least 1 before destroying it.
        // Higher values are also fine.
        let res = dispatch_semaphore_signal(sem.ptr);
        assert_eq!(res, 0);
        let res = dispatch_semaphore_signal(sem.ptr);
        assert_eq!(res, 0);

        dispatch_release(sem.ptr);
    }
}
