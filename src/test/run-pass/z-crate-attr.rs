// This test checks if an unstable feature is enabled with the -Zcrate-attr=feature(foo) flag. If
// the exact feature used here is causing problems feel free to replace it with another
// perma-unstable feature.

// compile-flags: -Zcrate-attr=feature(abi_unadjusted)

#![allow(dead_code)]

extern "unadjusted" fn foo() {}

fn main() {}
