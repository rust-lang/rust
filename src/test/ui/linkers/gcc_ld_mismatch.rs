// check-fail
// compile-flags: -Zgcc-ld=lld -Clinker-flavor=gcc:not_lld -Zunstable-options

// Test ensuring that until the unstable flag is removed (if ever), if both the linker-flavor and
// `gcc-ld` flags are used, they ask for the same linker.

fn main() {}
