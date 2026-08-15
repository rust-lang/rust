//@ compile-flags: -Copt-level=3
//@ only-x86_64

#![crate_type = "rlib"]
#![feature(asm_goto_with_outputs)]

use std::arch::asm;

// CHECK-LABEL: @asm_goto
#[no_mangle]
pub unsafe fn asm_goto() {
    // CHECK: callbr void asm sideeffect alignstack inteldialect "
    // CHECK-NEXT: to label %[[FALLTHROUGHBB:[a-b0-9]+]] [label %[[JUMPBB:[a-b0-9]+]]]
    asm!("jmp {}", label {});
}

// CHECK-LABEL: @asm_goto_with_outputs
#[no_mangle]
pub unsafe fn asm_goto_with_outputs() -> u64 {
    let out: u64;
    // CHECK: [[RES:%[0-9]+]] = callbr i64 asm sideeffect alignstack inteldialect "
    // CHECK-NEXT: to label %[[FALLTHROUGHBB:[a-b0-9]+]] [label %[[JUMPBB:[a-b0-9]+]]]
    asm!("{} /* {} */", out(reg) out, label { return 1; });
    // CHECK: [[JUMPBB]]:
    // CHECK-NEXT: [[RET:%.+]] = phi i64 [ [[RES]], %[[FALLTHROUGHBB]] ], [ 1, %start ]
    // CHECK-NEXT: ret i64 [[RET]]
    out
}

// CHECK-LABEL: @asm_goto_with_outputs_use_in_label
#[no_mangle]
pub unsafe fn asm_goto_with_outputs_use_in_label() -> u64 {
    let out: u64;
    // CHECK: [[RES:%[0-9]+]] = callbr i64 asm sideeffect alignstack inteldialect "
    // CHECK-NEXT: to label %[[FALLTHROUGHBB:[a-b0-9]+]] [label %[[JUMPBB:[a-b0-9]+]]]
    asm!("{} /* {} */", out(reg) out, label { return out; });
    // CHECK: [[JUMPBB]]:
    // CHECK-NEXT: [[RET:%.+]] = phi i64 [ 1, %[[FALLTHROUGHBB]] ], [ [[RES]], %start ]
    // CHECK-NEXT: ret i64 [[RET]]
    1
}

// CHECK-LABEL: @asm_goto_noreturn
#[no_mangle]
pub unsafe fn asm_goto_noreturn() -> u64 {
    // CHECK: callbr void asm sideeffect alignstack inteldialect "
    // CHECK-NEXT: to label %unreachable [label %[[JUMPBB:[a-b0-9]+]]]
    asm!("jmp {}", label { return 1; }, options(noreturn));
    // CHECK: [[JUMPBB]]:
    // CHECK-NEXT: ret i64 1
}

// CHECK-LABEL: @asm_goto_noreturn_with_outputs
#[no_mangle]
pub unsafe fn asm_goto_noreturn_with_outputs() -> u64 {
    let out: u64;
    // CHECK: [[RES:%[0-9]+]] = callbr i64 asm sideeffect alignstack inteldialect "
    // CHECK-NEXT: to label %[[FALLTHROUGHBB:[a-b0-9]+]] [label %[[JUMPBB:[a-b0-9]+]]]
    asm!("mov {}, 1", "jmp {}", out(reg) out, label { return out; });
    // CHECK: [[JUMPBB]]:
    // CHECK-NEXT: ret i64 [[RES]]
    out
}
