//@ compile-flags: -Copt-level=0
//@ only-64bit

#![crate_type = "lib"]

use std::ops::ControlFlow;

pub enum Never {}

#[no_mangle]
pub fn make_unmake_result_never(x: i32) -> i32 {
    // CHECK-LABEL: define i32 @make_unmake_result_never(i32{{( signext)?}} %x)
    // CHECK: start:
    // CHECK-NEXT: ret i32 %x

    let y: Result<i32, Never> = Ok(x);
    let Ok(z) = y;
    z
}

#[no_mangle]
pub fn extract_control_flow_never(x: ControlFlow<&str, Never>) -> &str {
    // CHECK-LABEL: define { ptr, i64 } @extract_control_flow_never(ptr align 1 %x.0, i64 %x.1)
    // CHECK: start:
    // CHECK-NEXT: %[[P0:.+]] = insertvalue { ptr, i64 } poison, ptr %x.0, 0
    // CHECK-NEXT: %[[P1:.+]] = insertvalue { ptr, i64 } %[[P0]], i64 %x.1, 1
    // CHECK-NEXT: ret { ptr, i64 } %[[P1]]

    let ControlFlow::Break(s) = x;
    s
}
