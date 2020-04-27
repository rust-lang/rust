#![feature(rustc_private)]

extern crate libc;

// error-pattern: scalar size mismatch
fn main() {
    extern "C" {
        fn malloc(size: u32) -> *mut std::ffi::c_void;
    }

    unsafe {
        let p1 = malloc(42);
        libc::free(p1);
    };
}
