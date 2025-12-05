//@ run-pass
//@ compile-flags:-Zforce-unstable-if-unmarked

#[deprecated] // should work even with -Zforce-unstable-if-unmarked
fn main() {}
