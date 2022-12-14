// compile-flags:-C panic=unwind
// no-prefer-dynamic

#![crate_type = "rlib"]
#![no_std]
#![feature(c_unwind)]

extern "C-unwind" fn foo() {}

fn bar() {
    let ptr: extern "C-unwind" fn() = foo;
    ptr();
}
