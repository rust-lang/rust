fn main() {
    extern "C" {
        // Use the wrong type (ie. not `i32`) for the `c` argument.
        fn memchr(s: *const std::ffi::c_void, c: u8, n: usize) -> *mut std::ffi::c_void;
    }

    unsafe {
        memchr(std::ptr::null(), 0, 0); //~ ERROR: Undefined Behavior: scalar size mismatch
    };
}
