#![warn(clippy::invalid_atomic_ordering)]

use std::sync::atomic::{AtomicBool, Ordering};

fn main() {
    let x = AtomicBool::new(true);

    // Allowed load ordering modes
    let _ = x.load(Ordering::Acquire);
    let _ = x.load(Ordering::SeqCst);
    let _ = x.load(Ordering::Relaxed);

    // Disallowed load ordering modes
    let _ = x.load(Ordering::Release);
    let _ = x.load(Ordering::AcqRel);

    // Allowed store ordering modes
    x.store(false, Ordering::Release);
    x.store(false, Ordering::SeqCst);
    x.store(false, Ordering::Relaxed);

    // Disallowed store ordering modes
    x.store(false, Ordering::Acquire);
    x.store(false, Ordering::AcqRel);
}
