// Checks that when we use `-Zself-profile-events=args`, it is possible to pretty print paths
// using `trimmed_def_paths` even without producing diagnostics.
//
// Issue: <https://github.com/rust-lang/rust/issues/144457>.

//@ compile-flags: -Zself-profile={{build-base}} -Zself-profile-events=args
//@ build-pass

use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;

fn main() {
    AtomicUsize::new(0).load(Relaxed);
}
