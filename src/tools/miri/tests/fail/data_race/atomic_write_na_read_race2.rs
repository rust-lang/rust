// We want to control preemption here. Stacked borrows interferes by having its own accesses.
//@compile-flags: -Zmiri-preemption-rate=0 -Zmiri-disable-stacked-borrows
// Avoid accidental synchronization via address reuse inside `thread::spawn`.
//@compile-flags: -Zmiri-address-reuse-cross-thread-rate=0

use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::spawn;

#[derive(Copy, Clone)]
struct EvilSend<T>(pub T);

unsafe impl<T> Send for EvilSend<T> {}
unsafe impl<T> Sync for EvilSend<T> {}

pub fn main() {
    let mut a = AtomicUsize::new(0);
    let b = &mut a as *mut AtomicUsize;
    let c = EvilSend(b);
    unsafe {
        let j1 = spawn(move || {
            let c = c; // avoid field capturing
            let _val = *(c.0 as *mut usize);
        });

        let j2 = spawn(move || {
            let c = c; // avoid field capturing
            (&*c.0).store(32, Ordering::SeqCst); //~ ERROR: Data race detected between (1) non-atomic read on thread `unnamed-1` and (2) atomic store on thread `unnamed-2`
        });

        j1.join().unwrap();
        j2.join().unwrap();
    }
}
