// ignore-windows: Concurrency on Windows is not supported yet.

//! Check if Rust once statics are working. The test taken from the Rust
//! documentation.

use std::sync::Once;
use std::thread;

static mut VAL: usize = 0;
static INIT: Once = Once::new();

fn get_cached_val() -> usize {
    unsafe {
        INIT.call_once(|| {
            VAL = expensive_computation();
        });
        VAL
    }
}

fn expensive_computation() -> usize {
    let mut i = 1;
    let mut c = 1;
    while i < 10000 {
        i *= c;
        c += 1;
    }
    i
}

fn main() {
    let handles: Vec<_> = (0..10)
        .map(|_| {
            thread::spawn(|| {
                thread::yield_now();
                let val = get_cached_val();
                assert_eq!(val, 40320);
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }
}
