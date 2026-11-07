//@ run-pass
//! Test that #[derive(Copy, Clone)] produces a shallow copy
//! even when a member violates RFC 1521

use std::sync::atomic::{AtomicBool, Ordering};

/// A struct that pretends to be Copy, but actually does something
/// in its Clone impl
#[derive(Copy)]
struct Liar;

/// Static cooperating with the rogue Clone impl
static CLONED: AtomicBool = AtomicBool::new(false);

impl Clone for Liar {
    fn clone(&self) -> Self {
        // this makes Clone vs Copy observable
        CLONED.store(true, Ordering::SeqCst);

        *self
    }
}

/// This struct is actually Copy... at least, it thinks it is!
#[derive(Copy, Clone)]
struct Innocent(#[allow(dead_code)] Liar);

impl Innocent {
    fn new() -> Self {
        Innocent(Liar)
    }
}

fn main() {
    let _ = Innocent::new().clone();
    // if Innocent was byte-for-byte copied, CLONED will still be false
    assert!(!CLONED.load(Ordering::SeqCst));
}
