// compile-flags: -O

// ignore-asmjs

#![feature(asm)]
#![crate_type = "lib"]

// Check that inline assembly expressions without any outputs
// are marked as having side effects / being volatile

// CHECK-LABEL: @assembly
#[no_mangle]
pub fn assembly() {
    unsafe { asm!("") }
// CHECK: tail call void asm sideeffect "", {{.*}}
}
