//@ compile-flags: -Z public-api-hash

#![crate_name = "dep"]
#![crate_type = "rlib"]

mod private;

pub fn call_private() {
    private::print();
}

#[cfg(any(cpass3))]
pub fn new_public() {}
