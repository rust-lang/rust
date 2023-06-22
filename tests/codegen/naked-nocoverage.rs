// Checks that naked functions are not instrumented by -Cinstrument-coverage.
// Regression test for issue #105170.
//
// needs-asm-support
// needs-profiler-support
// compile-flags: -Cinstrument-coverage
#![crate_type = "lib"]
#![feature(naked_functions)]
use std::arch::asm;

#[naked]
#[no_mangle]
pub unsafe extern "C" fn f() {
    // CHECK:       define {{(dso_local )?}}void @f()
    // CHECK-NEXT:  start:
    // CHECK-NEXT:    call void asm
    // CHECK-NEXT:    unreachable
    asm!("", options(noreturn));
}
