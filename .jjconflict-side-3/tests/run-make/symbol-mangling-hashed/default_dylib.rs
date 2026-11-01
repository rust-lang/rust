#![crate_type = "dylib"]

extern crate hashed_dylib;
extern crate hashed_rlib;

pub fn ddhello() {
    hashed_rlib::hrhello();
    hashed_dylib::hdhello();
}
