#![crate_type = "dylib"]

extern crate foo;

#[no_mangle]
pub extern "C" fn bar() {
    foo::foo();
}
