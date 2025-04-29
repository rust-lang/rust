//@compile-flags: -Zmiri-disable-isolation -Zmiri-deterministic-concurrency
//@ignore-target: windows # No libc env support on Windows

use std::ffi::CStr;
use std::thread;

fn main() {
    unsafe {
        thread::spawn(|| {
            // Access the environment in another thread without taking the env lock
            let s = libc::getenv("MIRI_ENV_VAR_TEST\0".as_ptr().cast());
            if s.is_null() {
                panic!("null");
            }
            let _s = String::from_utf8_lossy(CStr::from_ptr(s).to_bytes());
        });
        thread::yield_now();
        // After the main thread exits, env vars will be cleaned up -- but because we have not *joined*
        // the other thread, those accesses technically race with those in the other thread.
        // We don't want to emit an error here, though.
    }
}
