// run-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

unsafe extern fn foo() {}
unsafe extern "C" fn bar() {}

fn main() {
    let _a: unsafe extern fn() = foo;
    let _a: unsafe extern "C" fn() = foo;
}
