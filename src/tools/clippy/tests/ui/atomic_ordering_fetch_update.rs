#![warn(clippy::invalid_atomic_ordering)]

use std::sync::atomic::{AtomicIsize, Ordering};

fn main() {
    // `fetch_update` testing
    let x = AtomicIsize::new(0);

    // Allowed ordering combos
    let _ = x.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |old| Some(old + 1));
    let _ = x.fetch_update(Ordering::Acquire, Ordering::Acquire, |old| Some(old + 1));
    let _ = x.fetch_update(Ordering::Acquire, Ordering::Relaxed, |old| Some(old + 1));
    let _ = x.fetch_update(Ordering::Release, Ordering::Relaxed, |old| Some(old + 1));
    let _ = x.fetch_update(Ordering::AcqRel, Ordering::Acquire, |old| Some(old + 1));
    let _ = x.fetch_update(Ordering::AcqRel, Ordering::Relaxed, |old| Some(old + 1));
    let _ = x.fetch_update(Ordering::SeqCst, Ordering::Relaxed, |old| Some(old + 1));
    let _ = x.fetch_update(Ordering::SeqCst, Ordering::Acquire, |old| Some(old + 1));
    let _ = x.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |old| Some(old + 1));

    // AcqRel is always forbidden as a failure ordering
    let _ = x.fetch_update(Ordering::Relaxed, Ordering::AcqRel, |old| Some(old + 1));
    let _ = x.fetch_update(Ordering::Acquire, Ordering::AcqRel, |old| Some(old + 1));
    let _ = x.fetch_update(Ordering::Release, Ordering::AcqRel, |old| Some(old + 1));
    let _ = x.fetch_update(Ordering::AcqRel, Ordering::AcqRel, |old| Some(old + 1));
    let _ = x.fetch_update(Ordering::SeqCst, Ordering::AcqRel, |old| Some(old + 1));

    // Release is always forbidden as a failure ordering
    let _ = x.fetch_update(Ordering::Relaxed, Ordering::Release, |old| Some(old + 1));
    let _ = x.fetch_update(Ordering::Acquire, Ordering::Release, |old| Some(old + 1));
    let _ = x.fetch_update(Ordering::Release, Ordering::Release, |old| Some(old + 1));
    let _ = x.fetch_update(Ordering::AcqRel, Ordering::Release, |old| Some(old + 1));
    let _ = x.fetch_update(Ordering::SeqCst, Ordering::Release, |old| Some(old + 1));

    // Release success order forbids failure order of Acquire or SeqCst
    let _ = x.fetch_update(Ordering::Release, Ordering::Acquire, |old| Some(old + 1));
    let _ = x.fetch_update(Ordering::Release, Ordering::SeqCst, |old| Some(old + 1));

    // Relaxed success order also forbids failure order of Acquire or SeqCst
    let _ = x.fetch_update(Ordering::Relaxed, Ordering::SeqCst, |old| Some(old + 1));
    let _ = x.fetch_update(Ordering::Relaxed, Ordering::Acquire, |old| Some(old + 1));

    // Acquire/AcqRel forbids failure order of SeqCst
    let _ = x.fetch_update(Ordering::Acquire, Ordering::SeqCst, |old| Some(old + 1));
    let _ = x.fetch_update(Ordering::AcqRel, Ordering::SeqCst, |old| Some(old + 1));
}
