// compile-flags: -O

#![crate_type = "lib"]

use std::arch::asm;

#[no_mangle]
// CHECK-LABEL: @reserve_extend(
pub fn reserve_extend(v: &mut Vec<u8>, s: &[u8]) {
    // CHECK: do_reserve_and_handle
    // CHECK: "nop"
    // CHECK-NOT: do_reserve_and_handle
    // CHECK: ret
    v.reserve(s.len());
    unsafe {
        asm!("nop"); // Used to make the check logic easier
    }
    v.extend_from_slice(s);
}

#[no_mangle]
// CHECK-LABEL: @reserve_exact_extend(
pub fn reserve_exact_extend(v: &mut Vec<u8>, s: &[u8]) {
    // CHECK: finish_grow
    // CHECK: "nop"
    // CHECK-NOT: finish_grow
    // CHECK: ret
    v.reserve_exact(s.len());
    unsafe {
        asm!("nop"); // Used to make the check logic easier
    }
    v.extend_from_slice(s);
}

#[no_mangle]
// CHECK-LABEL: @try_reserve_extend(
pub fn try_reserve_extend(v: &mut Vec<u8>, s: &[u8]) {
    // CHECK: finish_grow
    // CHECK: "nop"
    // CHECK-NOT: finish_grow
    // CHECK: ret
    v.try_reserve(s.len());
    unsafe {
        asm!("nop"); // Used to make the check logic easier
    }
    v.extend_from_slice(s);
}

#[no_mangle]
// CHECK-LABEL: @try_reserve_exact_extend(
pub fn try_reserve_exact_extend(v: &mut Vec<u8>, s: &[u8]) {
    // CHECK: finish_grow
    // CHECK: "nop"
    // CHECK-NOT: finish_grow
    // CHECK: ret
    v.try_reserve_exact(s.len());
    unsafe {
        asm!("nop"); // Used to make the check logic easier
    }
    v.extend_from_slice(s);
}
