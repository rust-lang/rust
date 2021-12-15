// min-llvm-version: 13.0.0
// compile-flags: -O
// only-x86_64

#![crate_type = "rlib"]
#![feature(asm_unwind)]

use std::arch::asm;

#[no_mangle]
pub extern "C" fn panicky() {}

struct Foo;

impl Drop for Foo {
    fn drop(&mut self) {
        println!();
    }
}

// CHECK-LABEL: @may_unwind
#[no_mangle]
pub unsafe fn may_unwind() {
    let _m = Foo;
    // CHECK: invoke void asm sideeffect alignstack inteldialect unwind ""
    asm!("", options(may_unwind));
}
