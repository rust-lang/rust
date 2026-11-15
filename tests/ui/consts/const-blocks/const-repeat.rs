//@ run-pass

use std::sync::atomic::{AtomicUsize, Ordering};

// Repeating a *constant* of non-Copy type (not just a constant expression) is already stable.

const EMPTY: Vec<i32> = Vec::new();

pub fn bar() -> [Vec<i32>; 2] {
    [EMPTY; 2]
}

static NUM_DROPS: AtomicUsize = AtomicUsize::new(0);

struct NoisyDrop;

impl Drop for NoisyDrop {
    fn drop(&mut self) {
        NUM_DROPS.fetch_add(1, Ordering::Relaxed);
    }
}

const NOISY: NoisyDrop = NoisyDrop;

fn main() {
    let _x = bar();

    // Make sure the destructor gets called even for empty arrays.
    // `[CONST; N]` should instantiate the const.
    let _x = [NOISY; 0];

    assert_eq!(NUM_DROPS.load(Ordering::Relaxed), 1);
}
