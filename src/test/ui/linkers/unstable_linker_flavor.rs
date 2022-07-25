// check-fail
// compile-flags: -C linker-flavor=gcc:lld

// Test ensuring that the unstable `gcc:lld` value of the stable `-C linker-flavor` flag requires
// using `-Z unstable options`

fn main() {}
