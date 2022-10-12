// We want to control preemption here.
//@compile-flags: -Zmiri-preemption-rate=0

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
            *(c.0 as *mut usize) = 32;
        });

        let j2 = spawn(move || {
            (&*c.0).load(Ordering::SeqCst) //~ ERROR: Data race detected between Atomic Load on thread `<unnamed>` and Write on thread `<unnamed>`
        });

        j1.join().unwrap();
        j2.join().unwrap();
    }
}
