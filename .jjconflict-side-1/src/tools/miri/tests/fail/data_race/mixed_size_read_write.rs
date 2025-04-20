//@compile-flags:-Zmiri-deterministic-concurrency
// Two revisions, depending on which access goes first.
//@revisions: read_write write_read

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
            if cfg!(read_write) {
                // Let the other one go first.
                thread::yield_now();
            }
            a16.store(1, Ordering::SeqCst);
            //~[read_write]^ ERROR: Race condition detected between (1) 1-byte atomic load on thread `unnamed-2` and (2) 2-byte atomic store on thread `unnamed-1`
        });
        s.spawn(|| {
            if cfg!(write_read) {
                // Let the other one go first.
                thread::yield_now();
            }
            a8[0].load(Ordering::SeqCst);
            //~[write_read]^ ERROR: Race condition detected between (1) 2-byte atomic store on thread `unnamed-1` and (2) 1-byte atomic load on thread `unnamed-2`
        });
    });
}
