// min-llvm-version: 13.0.0
// compile-flags: -O

#![crate_type = "rlib"]
#![feature(asm, asm_unwind)]

#[no_mangle]
pub extern "C" fn panicky() {}

struct Foo;

impl Drop for Foo {
    fn drop(&mut self) {
        println!();
    }
}

// CHECK-LABEL: @may_unwind
// CHECK: invoke void asm sideeffect alignstack unwind
#[no_mangle]
pub unsafe fn may_unwind() {
    let _m = Foo;
    asm!("", options(may_unwind));
}
