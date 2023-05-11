//@compile-flags: -Zmiri-disable-isolation -Zmiri-preemption-rate=0
//@ignore-target-windows: No libc on Windows

use std::ffi::CStr;
use std::ffi::CString;
use std::thread;

fn main() {
    unsafe {
        thread::spawn(|| {
            // Access the environment in another thread without taking the env lock
            let k = CString::new("MIRI_ENV_VAR_TEST".as_bytes()).unwrap();
            let s = libc::getenv(k.as_ptr()) as *const libc::c_char;
            if s.is_null() {
                panic!("null");
            }
            let _s = String::from_utf8_lossy(CStr::from_ptr(s).to_bytes());
        });
        thread::yield_now();
        // After the main thread exits, env vars will be cleaned up -- but because we have not *joined*
        // the other thread, those accesses technically race with those in the other thread.
    }
}
