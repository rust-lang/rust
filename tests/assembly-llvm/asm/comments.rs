//@ assembly-output: emit-asm
//@ only-x86_64
// Check that comments in assembly get passed

#![crate_type = "lib"]

// CHECK-LABEL: test_comments:
#[no_mangle]
pub fn test_comments() {
    // CHECK: example comment
    unsafe { core::arch::asm!("nop // example comment") };
}
