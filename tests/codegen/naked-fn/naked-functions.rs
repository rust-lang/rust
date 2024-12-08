//@ compile-flags: -C no-prepopulate-passes -Copt-level=0
//@ needs-asm-support
//@ only-x86_64

#![crate_type = "lib"]
#![feature(naked_functions)]
use std::arch::naked_asm;

// CHECK: Function Attrs: naked
// CHECK-NEXT: define{{.*}}void @naked_empty()
#[no_mangle]
#[naked]
pub unsafe extern "C" fn naked_empty() {
    // CHECK-NEXT: {{.+}}:
    // CHECK-NEXT: call void asm
    // CHECK-NEXT: unreachable
    naked_asm!("ret");
}

// CHECK: Function Attrs: naked
// CHECK-NEXT: define{{.*}}i{{[0-9]+}} @naked_with_args_and_return(i64 %0, i64 %1)
#[no_mangle]
#[naked]
pub unsafe extern "C" fn naked_with_args_and_return(a: isize, b: isize) -> isize {
    // CHECK-NEXT: {{.+}}:
    // CHECK-NEXT: call void asm
    // CHECK-NEXT: unreachable
    naked_asm!("lea rax, [rdi + rsi]", "ret");
}
