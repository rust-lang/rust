//@only-target: windows # Uses win32 api functions
// We are making scheduler assumptions here.
//@compile-flags: -Zmiri-preemption-rate=0

use std::os::windows::io::IntoRawHandle;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use windows_sys::Win32::Foundation::{HANDLE, WAIT_OBJECT_0};
use windows_sys::Win32::System::Threading::{INFINITE, WaitForSingleObject};

#[derive(Copy, Clone)]
struct UnsafeSendWrapper<T>(T);
unsafe impl<T> Send for UnsafeSendWrapper<T> {}

fn main() {
    static FLAG: AtomicBool = AtomicBool::new(false);

    let blocker = thread::spawn(|| {
        while !FLAG.load(Ordering::Relaxed) {
            thread::yield_now();
        }
    })
    .into_raw_handle();
    let blocker = UnsafeSendWrapper(blocker as HANDLE);

    let waiter = move || unsafe {
        let blocker = blocker; // circumvent per-field capturing
        assert_eq!(WaitForSingleObject(blocker.0, INFINITE), WAIT_OBJECT_0);
    };

    let waiter1 = thread::spawn(waiter);
    let waiter2 = thread::spawn(waiter);

    // this yield ensures `waiter1` & `waiter2` are blocked on `blocker` by this point
    thread::yield_now();

    FLAG.store(true, Ordering::Relaxed);

    waiter1.join().unwrap();
    waiter2.join().unwrap();
}
