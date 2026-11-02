#![allow(invalid_runtime_symbol_definitions)]

fn main() {
    extern "C" {
        fn malloc() -> *mut std::ffi::c_void;
    }

    unsafe {
        let _ = malloc(); //~ ERROR: takes 1 argument, but 0 arguments were given
    };
}
