#![crate_type="dylib"]

extern crate a_rlib;
extern crate a_dylib;

pub fn hello() {
    a_rlib::hello();
    a_dylib::hello();
}
