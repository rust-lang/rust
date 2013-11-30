#[crate_type = "dylib"];
extern mod rlib;

pub fn dylib() { rlib::rlib() }
