//@ assembly-output: emit-asm
// Check that comments in assembly get passed

#![crate_type = "lib"]

// CHECK-LABEL: test_comments:
#[no_mangle]
pub fn test_comments() {
    // CHECK: example comment
    unsafe { core::arch::asm!("nop // example comment") };
}
