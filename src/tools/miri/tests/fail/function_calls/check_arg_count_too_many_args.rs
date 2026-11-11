#![allow(invalid_runtime_symbol_definitions)]

fn main() {
    extern "C" {
        fn malloc(_: i32, _: i32) -> *mut std::ffi::c_void;
    }

    unsafe {
        let _ = malloc(1, 2); //~ ERROR: takes 1 argument, but 2 arguments were given
    };
}
