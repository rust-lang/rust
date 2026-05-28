//@compile-flags:-Zmiri-deterministic-concurrency
// Two variants: the atomic store matches the size of the first or second atomic load.
//@revisions: match_first_load match_second_load

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
            a16.load(Ordering::SeqCst);
        });
        s.spawn(|| {
            a8[0].load(Ordering::SeqCst);
        });
        s.spawn(|| {
            thread::yield_now(); // make sure this happens last
            if cfg!(match_first_load) {
                a16.store(0, Ordering::SeqCst);
                //~[match_first_load]^ ERROR: Race condition detected between (1) multiple differently-sized atomic loads, including one load on thread `unnamed-1` and (2) 2-byte atomic store on thread `unnamed-3`
            } else {
                a8[0].store(0, Ordering::SeqCst);
                //~[match_second_load]^ ERROR: Race condition detected between (1) multiple differently-sized atomic loads, including one load on thread `unnamed-1` and (2) 1-byte atomic store on thread `unnamed-3`
            }
        });
    });
}
