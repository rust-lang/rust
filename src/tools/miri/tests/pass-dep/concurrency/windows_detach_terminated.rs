//@only-target: windows # Uses win32 api functions
// We are making scheduler assumptions here.
//@compile-flags: -Zmiri-deterministic-concurrency
//@run-native

use std::os::windows::io::IntoRawHandle;
use std::thread;

use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};

fn yield_now() {
    if cfg!(miri) {
        // We control the scheduler so we know this is enough.
        thread::yield_now();
    } else {
        // Without scheduler control, we hope that sleeping a bit is enough.
        thread::sleep(std::time::Duration::from_millis(20));
    }
}

fn main() {
    let thread = thread::spawn(|| {}).into_raw_handle() as HANDLE;

    // this yield ensures that `thread` is terminated by this point
    yield_now();

    unsafe {
        assert_ne!(CloseHandle(thread), 0);
    }
}
