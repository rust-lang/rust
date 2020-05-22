#![warn(clippy::invalid_atomic_ordering)]

use std::sync::atomic::{AtomicU16, AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering};

fn main() {
    // `AtomicU8` test cases
    let x = AtomicU8::new(0);

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

    // `AtomicU16` test cases
    let x = AtomicU16::new(0);

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

    // `AtomicU32` test cases
    let x = AtomicU32::new(0);

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

    // `AtomicU64` test cases
    let x = AtomicU64::new(0);

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

    // `AtomicUsize` test cases
    let x = AtomicUsize::new(0);

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
