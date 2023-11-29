// compile-flags: -Zforce-unstable-if-unmarked

// This emulates a dep-of-std (eg hashbrown), that has const functions it
// cannot mark as stable, and is build with force-unstable-if-unmarked.

pub const fn do_something_else() {}
