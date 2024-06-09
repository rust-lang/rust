//@ compile-flags: -O
// This regress since Rust version 1.72.
//@ min-llvm-version: 18.1.4

#![crate_type = "lib"]

use std::convert::TryInto;

const N: usize = 24;

// CHECK-LABEL: @example
// CHECK-NOT: unwrap_failed
#[no_mangle]
pub fn example(a: Vec<u8>) -> u8 {
    if a.len() != 32 {
        return 0;
    }

    let a: [u8; 32] = a.try_into().unwrap();

    a[15] + a[N]
}
