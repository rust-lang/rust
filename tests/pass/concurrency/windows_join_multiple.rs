//@only-target-windows: Uses win32 api functions
// We are making scheduler assumptions here.
//@compile-flags: -Zmiri-preemption-rate=0

use std::os::windows::io::IntoRawHandle;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

extern "system" {
    fn WaitForSingleObject(handle: usize, timeout: u32) -> u32;
}

const INFINITE: u32 = u32::MAX;

fn main() {
    static FLAG: AtomicBool = AtomicBool::new(false);

    let blocker = thread::spawn(|| {
        while !FLAG.load(Ordering::Relaxed) {
            thread::yield_now();
        }
    })
    .into_raw_handle() as usize;

    let waiter = move || {
        unsafe {
            assert_eq!(WaitForSingleObject(blocker, INFINITE), 0);
        }
    };

    let waiter1 = thread::spawn(waiter);
    let waiter2 = thread::spawn(waiter);

    // this yield ensures `waiter1` & `waiter2` are blocked on `blocker` by this point
    thread::yield_now();

    FLAG.store(true, Ordering::Relaxed);

    waiter1.join().unwrap();
    waiter2.join().unwrap();
}
