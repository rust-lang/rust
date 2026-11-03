//! Basic test for `--print=target-standard-library-support`

//@ compile-flags: --target x86_64-unknown-linux-gnu --print=target-standard-library-support -Z unstable-options
//@ needs-llvm-components: x86

//@ check-pass

fn main() {}
