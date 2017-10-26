//! Cache of run-time feature detection

use super::bit;
use std::sync::atomic::{AtomicUsize, Ordering};

/// This global variable is a bitset used to cache the features supported by
/// the
/// CPU.
static CACHE: AtomicUsize = AtomicUsize::new(::std::usize::MAX);

/// Test the `bit` of the storage. If the storage has not been initialized,
/// initializes it with the result of `f()`.
///
/// On its first invocation, it detects the CPU features and caches them in the
/// `FEATURES` global variable as an `AtomicUsize`.
///
/// It uses the `__Feature` variant to index into this variable as a bitset. If
/// the bit is set, the feature is enabled, and otherwise it is disabled.
///
/// PLEASE: do not use this, it is an implementation detail subject to change.
pub fn test<F>(bit: u32, f: F) -> bool
where
    F: FnOnce() -> usize,
{
    if CACHE.load(Ordering::Relaxed) == ::std::usize::MAX {
        CACHE.store(f(), Ordering::Relaxed);
    }
    bit::test(CACHE.load(Ordering::Relaxed), bit)
}
