//@ only-x86_64
//@ compile-flags: -C no-prepopulate-passes
#![feature(asm_goto_with_outputs)]
#![crate_type = "lib"]
use std::arch::asm;

// Regression test for #137867. Check that critical edges have been split before code generation,
// and so all stores to the asm output occur on disjoint paths without any of them jumping to
// another callbr label.
//
// CHECK-LABEL: @f(
// CHECK:        [[OUT:%.*]] = callbr i32 asm
// CHECK-NEXT:   to label %[[BB0:.*]] [label %[[BB1:.*]], label %[[BB2:.*]]],
// CHECK:       [[BB1]]:
// CHECK-NEXT:    store i32 [[OUT]], ptr %a
// CHECK-NEXT:    br label %[[BBR:.*]]
// CHECK:       [[BB2]]:
// CHECK-NEXT:    store i32 [[OUT]], ptr %a
// CHECK-NEXT:    br label %[[BBR]]
// CHECK:       [[BB0]]:
// CHECK-NEXT:    store i32 [[OUT]], ptr %a
// CHECK-NEXT:    br label %[[BBR]]
// CHECK:       [[BBR]]:
// CHECK-NEXT:    [[RET:%.*]] = load i32, ptr %a
// CHECK-NEXT:    ret i32 [[RET]]
#[unsafe(no_mangle)]
pub unsafe fn f(mut a: u32) -> u32 {
    asm!(
        "jmp {}
         jmp {}",
        label {},
        label {},
        inout("eax") a,
    );
    a
}
