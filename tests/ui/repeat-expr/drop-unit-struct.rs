//@ run-pass

use std::sync::atomic::{AtomicU8, Ordering};

static COUNTER: AtomicU8 = AtomicU8::new(0);

struct NoisyDrop;

impl Drop for NoisyDrop {
    fn drop(&mut self) {
        COUNTER.fetch_add(1, Ordering::Relaxed);
    }
}

fn main() {
    let _empty = [NoisyDrop; 0];
    assert_eq!(COUNTER.load(Ordering::Relaxed), 0);

    let many = [NoisyDrop; 42];
    assert_eq!(COUNTER.load(Ordering::Relaxed), 0);
    drop(many);
    assert_eq!(COUNTER.load(Ordering::Relaxed), 42);
}
