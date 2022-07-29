// run-pass
#![feature(let_else)]

use std::sync::atomic::{AtomicU8, Ordering};

static TRACKER: AtomicU8 = AtomicU8::new(0);

#[derive(Default)]
struct Droppy {
    inner: u32,
}

impl Drop for Droppy {
    fn drop(&mut self) {
        TRACKER.store(1, Ordering::Release);
        println!("I've been dropped");
    }
}

fn main() {
    assert_eq!(TRACKER.load(Ordering::Acquire), 0);
    let 0 = Droppy::default().inner else { return };
    assert_eq!(TRACKER.load(Ordering::Acquire), 1);
    println!("Should have dropped 👆");
}
