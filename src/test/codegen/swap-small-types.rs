// compile-flags: -O
// only-x86_64

#![crate_type = "lib"]

use std::mem::swap;

type RGB48 = [u16; 3];

// CHECK-LABEL: @swap_rgb48
#[no_mangle]
pub fn swap_rgb48(x: &mut RGB48, y: &mut RGB48) {
// CHECK-NOT: alloca
// CHECK: load i48
// CHECK: store i48
    swap(x, y)
}
