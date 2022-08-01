//@only-target-windows: Uses win32 api functions
// We are making scheduler assumptions here.
//@compile-flags: -Zmiri-preemption-rate=0

// On windows, joining main is not UB, but it will block a thread forever.

use std::thread;

extern "system" {
    fn WaitForSingleObject(handle: isize, timeout: u32) -> u32;
}

const INFINITE: u32 = u32::MAX;

// XXX HACK: This is how miri represents the handle for thread 0.
// This value can be "legitimately" obtained by using `GetCurrentThread` with `DuplicateHandle`
// but miri does not implement `DuplicateHandle` yet.
const MAIN_THREAD: isize = (2i32 << 30) as isize;

fn main() {
    thread::spawn(|| {
        unsafe {
            assert_eq!(WaitForSingleObject(MAIN_THREAD, INFINITE), 0); //~ ERROR: deadlock: the evaluated program deadlocked
        }
    })
    .join()
    .unwrap();
}
