// compile-flags: -O

#![crate_type = "lib"]

use std::arch::asm;

#[no_mangle]
// CHECK-LABEL: @reserve_push_str(
pub fn reserve_push_str(v: &mut String, s: &str) {
    // CHECK: do_reserve_and_handle
    // CHECK: "nop"
    // CHECK-NOT: do_reserve_and_handle
    // CHECK: ret
    v.reserve(s.len());
    unsafe {
        asm!("nop"); // Used to make the check logic easier
    }
    v.push_str(s);
}

#[no_mangle]
// CHECK-LABEL: @reserve_exact_push_str(
pub fn reserve_exact_push_str(v: &mut String, s: &str) {
    // CHECK: finish_grow
    // CHECK: "nop"
    // CHECK-NOT: finish_grow
    // CHECK: ret
    v.reserve_exact(s.len());
    unsafe {
        asm!("nop"); // Used to make the check logic easier
    }
    v.push_str(s);
}

#[no_mangle]
// CHECK-LABEL: @try_reserve_push_str(
pub fn try_reserve_push_str(v: &mut String, s: &str) {
    // CHECK: finish_grow
    // CHECK: "nop"
    // CHECK-NOT: finish_grow
    // CHECK: ret
    v.try_reserve(s.len());
    unsafe {
        asm!("nop"); // Used to make the check logic easier
    }
    v.push_str(s);
}

#[no_mangle]
// CHECK-LABEL: @try_reserve_exact_push_str(
pub fn try_reserve_exact_push_str(v: &mut String, s: &str) {
    // CHECK: finish_grow
    // CHECK: "nop"
    // CHECK-NOT: finish_grow
    // CHECK: ret
    v.try_reserve_exact(s.len());
    unsafe {
        asm!("nop"); // Used to make the check logic easier
    }
    v.push_str(s);
}
