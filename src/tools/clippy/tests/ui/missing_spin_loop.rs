//@run-rustfix
#![warn(clippy::missing_spin_loop)]
#![allow(clippy::bool_comparison)]
#![allow(unused_braces)]

use core::sync::atomic::{AtomicBool, Ordering};

fn main() {
    let b = AtomicBool::new(true);
    // Those should lint
    while b.load(Ordering::Acquire) {}

    while !b.load(Ordering::SeqCst) {}

    while b.load(Ordering::Acquire) == false {}

    while { true == b.load(Ordering::Acquire) } {}

    while b.compare_exchange(true, false, Ordering::Acquire, Ordering::Relaxed) != Ok(true) {}

    while Ok(false) != b.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed) {}

    // This is OK, as the body is not empty
    while b.load(Ordering::Acquire) {
        std::hint::spin_loop()
    }
    // TODO: also match on loop+match or while let
}
