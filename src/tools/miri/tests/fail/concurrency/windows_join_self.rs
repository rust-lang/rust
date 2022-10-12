//@only-target-windows: Uses win32 api functions
// We are making scheduler assumptions here.
//@compile-flags: -Zmiri-preemption-rate=0

// On windows, a thread joining itself is not UB, but it will deadlock.

use std::thread;

extern "system" {
    fn GetCurrentThread() -> usize;
    fn WaitForSingleObject(handle: usize, timeout: u32) -> u32;
}

const INFINITE: u32 = u32::MAX;

fn main() {
    thread::spawn(|| {
        unsafe {
            let native = GetCurrentThread();
            assert_eq!(WaitForSingleObject(native, INFINITE), 0); //~ ERROR: deadlock: the evaluated program deadlocked
        }
    })
    .join()
    .unwrap();
}
