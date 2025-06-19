//@only-target: windows # Uses win32 api functions
// We are making scheduler assumptions here.
//@compile-flags: -Zmiri-deterministic-concurrency
//@error-in-other-file: deadlock

// On windows, joining main is not UB, but it will block a thread forever.

use std::thread;

use windows_sys::Win32::Foundation::{HANDLE, WAIT_OBJECT_0};
use windows_sys::Win32::System::Threading::{INFINITE, WaitForSingleObject};

// XXX HACK: This is how miri represents the handle for thread 0.
// This value can be "legitimately" obtained by using `GetCurrentThread` with `DuplicateHandle`
// but miri does not implement `DuplicateHandle` yet.
const MAIN_THREAD: HANDLE = (2i32 << 29) as HANDLE;

fn main() {
    thread::spawn(|| {
        unsafe {
            assert_eq!(WaitForSingleObject(MAIN_THREAD, INFINITE), WAIT_OBJECT_0); //~ ERROR: deadlock
        }
    })
    .join()
    .unwrap();
}
