//@ run-pass
#![allow(dead_code, missing_abi)]

unsafe extern fn foo() {}
unsafe extern "C" fn bar() {}

fn main() {
    let _a: unsafe extern fn() = foo;
    let _a: unsafe extern "C" fn() = foo;
}
