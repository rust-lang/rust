// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(const_eval_select)]

use std::intrinsics::const_eval_select;

const fn foo(_: i32) -> i32 { 1 }

#[no_mangle]
pub fn hi(n: i32) -> i32 { n }

#[no_mangle]
pub unsafe fn hey() {
    // CHECK: call i32 @hi(i32
    const_eval_select((42,), foo, hi);
}
