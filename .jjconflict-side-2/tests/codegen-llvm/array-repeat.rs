//@ compile-flags: -Copt-level=3

#![crate_type = "lib"]

use std::array::repeat;

// CHECK-LABEL: @byte_repeat
#[no_mangle]
fn byte_repeat(b: u8) -> [u8; 1024] {
    // CHECK-NOT: alloca
    // CHECK-NOT: store
    // CHECK: memset
    repeat(b)
}
