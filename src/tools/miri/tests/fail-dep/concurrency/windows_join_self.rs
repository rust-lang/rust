//@only-target: windows # Uses win32 api functions
// We are making scheduler assumptions here.
//@compile-flags: -Zmiri-deterministic-concurrency
//@error-in-other-file: deadlock

// On windows, a thread joining itself is not UB, but it will deadlock.

use std::thread;

use windows_sys::Win32::Foundation::WAIT_OBJECT_0;
use windows_sys::Win32::System::Threading::{GetCurrentThread, INFINITE, WaitForSingleObject};

fn main() {
    thread::spawn(|| {
        unsafe {
            let native = GetCurrentThread();
            assert_eq!(WaitForSingleObject(native, INFINITE), WAIT_OBJECT_0); //~ ERROR: deadlock
        }
    })
    .join()
    .unwrap();
}
