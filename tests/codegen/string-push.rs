//@ compile-flags: -O
#![crate_type = "lib"]

use std::hint::black_box;

// CHECK-LABEL: @string_push_does_not_call_memcpy
#[no_mangle]
pub fn string_push_does_not_call_memcpy(s: &mut String, ch: char) {
    // CHECK-NOT: call void @llvm.memcpy
    s.push(ch);
}
