//@only-target: windows # Uses win32 api functions
//@error-in-other-file: Undefined Behavior: trying to join a detached thread

// Joining a detached thread is undefined behavior.

use std::os::windows::io::AsRawHandle;
use std::thread;

use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};

fn main() {
    let thread = thread::spawn(|| ());

    unsafe {
        assert_ne!(CloseHandle(thread.as_raw_handle() as HANDLE), 0);
    }

    thread.join().unwrap();
}
