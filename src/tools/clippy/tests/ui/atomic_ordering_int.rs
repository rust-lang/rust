#![warn(clippy::invalid_atomic_ordering)]

use std::sync::atomic::{AtomicI16, AtomicI32, AtomicI64, AtomicI8, AtomicIsize, Ordering};

fn main() {
    // `AtomicI8` test cases
    let x = AtomicI8::new(0);

    // Allowed load ordering modes
    let _ = x.load(Ordering::Acquire);
    let _ = x.load(Ordering::SeqCst);
    let _ = x.load(Ordering::Relaxed);

    // Disallowed load ordering modes
    let _ = x.load(Ordering::Release);
    let _ = x.load(Ordering::AcqRel);

    // Allowed store ordering modes
    x.store(1, Ordering::Release);
    x.store(1, Ordering::SeqCst);
    x.store(1, Ordering::Relaxed);

    // Disallowed store ordering modes
    x.store(1, Ordering::Acquire);
    x.store(1, Ordering::AcqRel);

    // `AtomicI16` test cases
    let x = AtomicI16::new(0);

    let _ = x.load(Ordering::Acquire);
    let _ = x.load(Ordering::SeqCst);
    let _ = x.load(Ordering::Relaxed);
    let _ = x.load(Ordering::Release);
    let _ = x.load(Ordering::AcqRel);

    x.store(1, Ordering::Release);
    x.store(1, Ordering::SeqCst);
    x.store(1, Ordering::Relaxed);
    x.store(1, Ordering::Acquire);
    x.store(1, Ordering::AcqRel);

    // `AtomicI32` test cases
    let x = AtomicI32::new(0);

    let _ = x.load(Ordering::Acquire);
    let _ = x.load(Ordering::SeqCst);
    let _ = x.load(Ordering::Relaxed);
    let _ = x.load(Ordering::Release);
    let _ = x.load(Ordering::AcqRel);

    x.store(1, Ordering::Release);
    x.store(1, Ordering::SeqCst);
    x.store(1, Ordering::Relaxed);
    x.store(1, Ordering::Acquire);
    x.store(1, Ordering::AcqRel);

    // `AtomicI64` test cases
    let x = AtomicI64::new(0);

    let _ = x.load(Ordering::Acquire);
    let _ = x.load(Ordering::SeqCst);
    let _ = x.load(Ordering::Relaxed);
    let _ = x.load(Ordering::Release);
    let _ = x.load(Ordering::AcqRel);

    x.store(1, Ordering::Release);
    x.store(1, Ordering::SeqCst);
    x.store(1, Ordering::Relaxed);
    x.store(1, Ordering::Acquire);
    x.store(1, Ordering::AcqRel);

    // `AtomicIsize` test cases
    let x = AtomicIsize::new(0);

    let _ = x.load(Ordering::Acquire);
    let _ = x.load(Ordering::SeqCst);
    let _ = x.load(Ordering::Relaxed);
    let _ = x.load(Ordering::Release);
    let _ = x.load(Ordering::AcqRel);

    x.store(1, Ordering::Release);
    x.store(1, Ordering::SeqCst);
    x.store(1, Ordering::Relaxed);
    x.store(1, Ordering::Acquire);
    x.store(1, Ordering::AcqRel);
}
