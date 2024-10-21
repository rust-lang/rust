//@ compile-flags: -O
//@ ignore-debug
#![crate_type = "lib"]

use std::collections::binary_heap::PeekMut;

// CHECK-LABEL: @peek_mut_pop
#[no_mangle]
pub fn peek_mut_pop(peek_mut: PeekMut<u32>) -> u32 {
    // CHECK-NOT: panic
    // CHECK-NOT: unwrap_failed
    PeekMut::pop(peek_mut)
}
