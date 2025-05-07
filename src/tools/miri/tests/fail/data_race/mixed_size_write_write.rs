//@compile-flags:-Zmiri-deterministic-concurrency
//@revisions: fst snd

use std::sync::atomic::{AtomicU8, AtomicU16, Ordering};
use std::thread;

fn convert(a: &AtomicU16) -> &[AtomicU8; 2] {
    unsafe { std::mem::transmute(a) }
}

// We can't allow mixed-size accesses; they are not possible in C++ and even
// Intel says you shouldn't do it.
fn main() {
    let a = AtomicU16::new(0);
    let a16 = &a;
    let a8 = convert(a16);

    thread::scope(|s| {
        s.spawn(|| {
            a16.store(1, Ordering::SeqCst);
        });
        s.spawn(|| {
            let idx = if cfg!(fst) { 0 } else { 1 };
            a8[idx].store(1, Ordering::SeqCst);
            //~^ ERROR: Race condition detected between (1) 2-byte atomic store on thread `unnamed-1` and (2) 1-byte atomic store on thread `unnamed-2`
        });
    });
}
