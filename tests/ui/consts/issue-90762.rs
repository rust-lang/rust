//@ run-pass
#![allow(unreachable_code)]

use std::sync::atomic::{AtomicBool, Ordering, AtomicUsize};

struct Print(usize);

impl Drop for Print {
    fn drop(&mut self) {
        println!("{}", self.0);
        FOO[self.0].store(true, Ordering::Relaxed);
        assert_eq!(BAR.fetch_sub(1, Ordering::Relaxed), self.0);
    }
}

const A: Print = Print(0);
const B: Print = Print(1);

static FOO: [AtomicBool; 3] =
    [AtomicBool::new(false), AtomicBool::new(false), AtomicBool::new(false)];
static BAR: AtomicUsize = AtomicUsize::new(2);

fn main() {
    loop {
        std::mem::forget(({ A }, B, Print(2), break));
    }
    for (i, b) in FOO.iter().enumerate() {
        assert!(b.load(Ordering::Relaxed), "{} not set", i);
    }
    assert_eq!(BAR.fetch_add(1, Ordering::Relaxed), usize::max_value());
}
