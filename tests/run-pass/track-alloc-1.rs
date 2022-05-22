// Ensure that tracking early allocations doesn't ICE Miri.
// Early allocations are probably part of the runtime and therefore uninteresting, but they
// shouldn't cause a crash.
// compile-flags: -Zmiri-track-alloc-id=1
fn main() {}
