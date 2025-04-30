// We want to control preemption here. Stacked borrows interferes by having its own accesses.
//@compile-flags: -Zmiri-deterministic-concurrency -Zmiri-disable-stacked-borrows

use std::thread::spawn;

#[derive(Copy, Clone)]
struct EvilSend<T>(pub T);

unsafe impl<T> Send for EvilSend<T> {}
unsafe impl<T> Sync for EvilSend<T> {}

pub fn main() {
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
            *c.0 = 64; //~ ERROR: Data race detected between (1) non-atomic write on thread `unnamed-1` and (2) non-atomic write on thread `unnamed-2`
        });

        j1.join().unwrap();
        j2.join().unwrap();
    }
}
