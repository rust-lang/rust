#![allow(invalid_runtime_symbol_definitions)]

fn main() {
    extern "C" {
        fn malloc() -> *mut std::ffi::c_void;
    }

    unsafe {
        let _ = malloc(); //~ ERROR: expected 1 arguments, found 0 arguments
    };
}
