#![crate_type = "dylib"]

extern crate a_dylib;
extern crate a_rlib;

pub fn hello() {
    a_rlib::hello();
    a_dylib::hello();
}
