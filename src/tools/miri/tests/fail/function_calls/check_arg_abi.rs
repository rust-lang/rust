fn main() {
    extern "Rust" {
        fn malloc(size: usize) -> *mut std::ffi::c_void;
    }

    unsafe {
        let _ = malloc(0); //~ ERROR: calling a function with calling convention "C" using caller calling convention "Rust"
    };
}
