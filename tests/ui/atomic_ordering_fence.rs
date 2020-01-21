#![warn(clippy::invalid_atomic_ordering)]

use std::sync::atomic::{compiler_fence, fence, Ordering};

fn main() {
    // Allowed fence ordering modes
    fence(Ordering::Acquire);
    fence(Ordering::Release);
    fence(Ordering::AcqRel);
    fence(Ordering::SeqCst);

    // Disallowed fence ordering modes
    fence(Ordering::Relaxed);

    compiler_fence(Ordering::Acquire);
    compiler_fence(Ordering::Release);
    compiler_fence(Ordering::AcqRel);
    compiler_fence(Ordering::SeqCst);
    compiler_fence(Ordering::Relaxed);
}
