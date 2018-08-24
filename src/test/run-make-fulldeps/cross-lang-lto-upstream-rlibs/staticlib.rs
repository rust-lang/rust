#![crate_type="staticlib"]

extern crate upstream;

#[no_mangle]
pub extern fn bar() {
    upstream::foo();
}
