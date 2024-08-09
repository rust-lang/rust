//@ run-pass
// This test verifies that temporaries created for `while`'s and `if`
// conditions are dropped after the condition is evaluated.

use std::sync::atomic::{AtomicUsize, Ordering};

struct Temporary;

static DROPPED: AtomicUsize = AtomicUsize::new(0);

impl Drop for Temporary {
    fn drop(&mut self) {
        DROPPED.fetch_add(1, Ordering::Relaxed);
    }
}

impl Temporary {
    fn do_stuff(&self) -> bool {true}
}

fn borrow() -> Box<Temporary> { Box::new(Temporary) }


pub fn main() {
    let mut i = 0;

    // This loop's condition
    // should call `Temporary`'s
    // `drop` 6 times.
    while borrow().do_stuff() {
        i += 1;
        assert_eq!(DROPPED.load(Ordering::Relaxed), i);
        if i > 5 {
            break;
        }
    }

    // This if condition should
    // call it 1 time
    if borrow().do_stuff() {
        assert_eq!(DROPPED.load(Ordering::Relaxed), i + 1);
    }
}
