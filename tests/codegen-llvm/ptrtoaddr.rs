//@ add-minicore
//@ compile-flags: -Copt-level=0 -Zllvm-ptrtoaddr
//@ only-64bit
#![crate_type = "lib"]
#![feature(no_core)]
#![no_core]
extern crate minicore;
use minicore::*;

// CHECK: define{{.*}}[[USIZE:i[0-9]+]] @ptr_to_int(ptr %p)
// CHECK: %_0 = ptrtoaddr ptr %p to [[USIZE]]
// CHECK-NEXT: ret [[USIZE]] %_0
#[no_mangle]
pub fn ptr_to_int(p: *mut u16) -> usize {
    unsafe { mem::transmute(p) }
}

// CHECK-LABEL: @check_float_from_pointer(
#[no_mangle]
pub unsafe fn check_float_from_pointer(x: *const ()) -> f64 {
    // CHECK-NOT: alloca
    // CHECK: %0 = ptrtoaddr ptr %x to i64
    // CHECK: %_0 = bitcast i64 %0 to double
    // CHECK: ret double %_0
    mem::transmute(x)
}
