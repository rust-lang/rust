#![crate_type="staticlib"]

extern crate upstream;

#[no_mangle]
pub extern "C" fn bar() {
    upstream::foo();
}
