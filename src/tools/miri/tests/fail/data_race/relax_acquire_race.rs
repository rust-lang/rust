// We want to control preemption here. Stacked borrows interferes by having its own accesses.
//@compile-flags: -Zmiri-deterministic-concurrency -Zmiri-disable-stacked-borrows

use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::spawn;

#[derive(Copy, Clone)]
struct EvilSend<T>(pub T);

unsafe impl<T> Send for EvilSend<T> {}
unsafe impl<T> Sync for EvilSend<T> {}

static SYNC: AtomicUsize = AtomicUsize::new(0);

pub fn main() {
    let mut a = 0u32;
    let b = &mut a as *mut u32;
    let c = EvilSend(b);

    // Note: this is scheduler-dependent
    // the operations need to occur in
    // order:
    //  1. store release : 1
    //  2. load acquire : 1
    //  3. store relaxed : 2
    //  4. load acquire : 2
    unsafe {
        let j1 = spawn(move || {
            let c = c; // avoid field capturing
            *c.0 = 1;
            SYNC.store(1, Ordering::Release);
        });

        let j2 = spawn(move || {
            if SYNC.load(Ordering::Acquire) == 1 {
                SYNC.store(2, Ordering::Relaxed);
            }
        });

        let j3 = spawn(move || {
            let c = c; // avoid field capturing
            if SYNC.load(Ordering::Acquire) == 2 {
                *c.0 //~ ERROR: Data race detected between (1) non-atomic write on thread `unnamed-1` and (2) non-atomic read on thread `unnamed-3`
            } else {
                0
            }
        });

        j1.join().unwrap();
        j2.join().unwrap();
        j3.join().unwrap();
    }
}
