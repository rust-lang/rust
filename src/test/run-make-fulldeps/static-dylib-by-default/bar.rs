#![crate_type = "dylib"]

extern crate foo;

#[no_mangle]
pub extern fn bar() {
    foo::foo();
}
