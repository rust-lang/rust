// We want to control preemption here. Stacked borrows interferes by having its own accesses.
//@compile-flags: -Zmiri-preemption-rate=0 -Zmiri-disable-stacked-borrows
// Avoid accidental synchronization via address reuse inside `thread::spawn`.
//@compile-flags: -Zmiri-address-reuse-cross-thread-rate=0

use std::thread::spawn;

#[derive(Copy, Clone)]
struct EvilSend<T>(pub T);

unsafe impl<T> Send for EvilSend<T> {}
unsafe impl<T> Sync for EvilSend<T> {}

pub fn main() {
    // Enable and then join with multiple threads.
    let t1 = spawn(|| ());
    let t2 = spawn(|| ());
    let t3 = spawn(|| ());
    let t4 = spawn(|| ());
    t1.join().unwrap();
    t2.join().unwrap();
    t3.join().unwrap();
    t4.join().unwrap();

    // Perform write-write data race detection.
    let mut a = 0u32;
    let b = &mut a as *mut u32;
    let c = EvilSend(b);
    unsafe {
        let j1 = spawn(move || {
            let c = c; // avoid field capturing
            *c.0 = 32;
        });

        let j2 = spawn(move || {
            let c = c; // avoid field capturing
            *c.0 = 64; //~ ERROR: Data race detected between (1) non-atomic write on thread `unnamed-5` and (2) non-atomic write on thread `unnamed-6`
        });

        j1.join().unwrap();
        j2.join().unwrap();
    }
}
