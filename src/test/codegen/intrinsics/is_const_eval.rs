// compile-flags: -C no-prepopulate-passes

#![crate_type = "lib"]
#![feature(core_intrinsics)]

use std::intrinsics::is_const_eval;

// CHECK-LABEL: @is_const_eval_test
#[no_mangle]
pub unsafe fn is_const_eval_test() -> bool {
    // CHECK: %0 = alloca i8, align 1
    // CHECK: store i8 0, i8* %0, align 1
    // CHECK: %2 = trunc i8 %1 to i1
    // CHECK: ret i1 %2
    is_const_eval()
}
