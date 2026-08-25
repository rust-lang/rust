use std::ffi::c_float;

extern "C" {
    pub fn sqrt(x: c_float) -> c_float;
}
