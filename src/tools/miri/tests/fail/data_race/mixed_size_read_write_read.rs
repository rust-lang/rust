//@compile-flags: -Zmiri-deterministic-concurrency
// A case that is not covered by `mixed_size_read_write`.
#![feature(ptr_as_ref_unchecked)]

use std::sync::atomic::*;
use std::thread;

fn main() {
    let data = AtomicI32::new(0);

    thread::scope(|s| {
        s.spawn(|| unsafe {
            let _val = (&raw const data).read();
            let _val = (&raw const data).cast::<AtomicI8>().as_ref_unchecked().compare_exchange(
                0,
                1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
            thread::yield_now();
            unreachable!();
        });
        s.spawn(|| {
            let _val = data.load(Ordering::Relaxed); //~ERROR: /Race condition detected between \(1\) 1-byte atomic store .* and \(2\) 4-byte atomic load/
        });
    });
}
