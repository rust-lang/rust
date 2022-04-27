// check-fail
// revisions: lld other
// [lld] compile-flags: -C linker-flavor=gcc:lld
// [other] compile-flags: -C linker-flavor=gcc:other

// Test ensuring that the unstable `gcc:*` values of the stable `-C linker-flavor` flag require
// using `-Z unstable options`

fn main() {}
