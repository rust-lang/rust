use std::ffi::c_float;

extern "C" {
    fn sqrt(x: c_float) -> c_float;
}
