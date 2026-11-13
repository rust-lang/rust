// Regression test for #165722: calling `resumable_breakpoint` at runtime ICEd in codegen.

#![crate_type = "lib"]
#![feature(breakpoint)]

// CHECK-LABEL: @call_resumable_breakpoint
#[no_mangle]
pub fn call_resumable_breakpoint() {
    // CHECK: call void @llvm.debugtrap()
    std::process::resumable_breakpoint();
}
