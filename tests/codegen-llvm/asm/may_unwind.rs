//@ compile-flags: -Copt-level=3
//@ only-x86_64

#![crate_type = "rlib"]
#![feature(asm_unwind)]

use std::arch::asm;

#[no_mangle]
pub extern "C" fn panicky() {}

struct Foo;

impl Drop for Foo {
    fn drop(&mut self) {
        println!();
    }
}

// CHECK-LABEL: @asm_may_unwind
#[no_mangle]
pub unsafe fn asm_may_unwind() {
    let _m = Foo;
    // CHECK: invoke void asm sideeffect alignstack inteldialect unwind ""
    asm!("", options(may_unwind));
}

// CHECK-LABEL: @asm_with_result_may_unwind
#[no_mangle]
pub unsafe fn asm_with_result_may_unwind() -> u64 {
    let _m = Foo;
    let res: u64;
    // CHECK: [[RES:%[0-9]+]] = invoke i64 asm sideeffect alignstack inteldialect unwind
    // CHECK-NEXT: to label %[[NORMALBB:[a-b0-9]+]]
    asm!("mov {}, 1", out(reg) res, options(may_unwind));
    // CHECK: [[NORMALBB]]:
    // CHECK: ret i64 [[RES:%[0-9]+]]
    res
}
