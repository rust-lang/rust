//@ compile-flags: -O

#![crate_type = "lib"]
#![feature(array_repeat)]

use std::array::repeat;

// CHECK-LABEL: @byte_repeat
#[no_mangle]
fn byte_repeat(b: u8) -> [u8; 1024] {
    // CHECK-NOT: alloca
    // CHECK-NOT: store
    // CHECK: memset
    repeat(b)
}
