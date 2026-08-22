#![allow(invalid_runtime_symbol_definitions)]

fn main() {
    extern "C" {
        fn malloc(_: i32, _: i32) -> *mut std::ffi::c_void;
    }

    unsafe {
        let _ = malloc(1, 2); //~ ERROR: expected 1 arguments, found 2 arguments
    };
}
