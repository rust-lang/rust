//@only-target: windows # Uses win32 api functions
// We are making scheduler assumptions here.
//@compile-flags: -Zmiri-deterministic-concurrency

use std::os::windows::io::IntoRawHandle;
use std::thread;

use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};

fn main() {
    let thread = thread::spawn(|| {}).into_raw_handle() as HANDLE;

    // this yield ensures that `thread` is terminated by this point
    thread::yield_now();

    unsafe {
        assert_ne!(CloseHandle(thread), 0);
    }
}
