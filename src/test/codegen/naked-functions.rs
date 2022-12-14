// compile-flags: -C no-prepopulate-passes
// needs-asm-support
// only-x86_64

#![crate_type = "lib"]
#![feature(naked_functions)]
use std::arch::asm;

// CHECK: Function Attrs: naked
// CHECK-NEXT: define{{.*}}void @naked_empty()
#[no_mangle]
#[naked]
pub unsafe extern "C" fn naked_empty() {
    // CHECK-NEXT: {{.+}}:
    // CHECK-NEXT: call void asm
    // CHECK-NEXT: unreachable
    asm!("ret",
         options(noreturn));
}

// CHECK: Function Attrs: naked
// CHECK-NEXT: define{{.*}}i{{[0-9]+}} @naked_with_args_and_return(i64 %a, i64 %b)
#[no_mangle]
#[naked]
pub unsafe extern "C" fn naked_with_args_and_return(a: isize, b: isize) -> isize {
    // CHECK-NEXT: {{.+}}:
    // CHECK-NEXT: call void asm
    // CHECK-NEXT: unreachable
    asm!("lea rax, [rdi + rsi]",
         "ret",
         options(noreturn));
}
