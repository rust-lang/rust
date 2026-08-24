#![allow(invalid_runtime_symbol_definitions)]

fn main() {
    extern "Rust" {
        fn malloc(size: usize) -> *mut std::ffi::c_void;
    }

    unsafe {
        let _ = malloc(0); //~ ERROR: has calling convention "C", but the caller is using calling convention "Rust"
    };
}
