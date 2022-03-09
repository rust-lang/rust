// run-pass
#![allow(unreachable_code)]

use std::sync::atomic::{AtomicBool, Ordering};

struct Print(usize);

impl Drop for Print {
    fn drop(&mut self) {
        FOO[self.0].store(true, Ordering::Relaxed);
    }
}

const A: Print = Print(0);
const B: Print = Print(1);

static FOO: [AtomicBool; 3] = [AtomicBool::new(false), AtomicBool::new(false), AtomicBool::new(false)];

fn main() {
    loop {
        std::mem::forget(({A}, B, Print(2), break));
    }
    for (i, b) in FOO.iter().enumerate() {
        assert!(b.load(Ordering::Relaxed), "{} not set", i);
    }
}