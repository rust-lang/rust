fn main() {
    extern "C" {
        // Use the wrong type(ie. not the pointer width) for the `size`
        // argument.
        #[cfg(target_pointer_width="64")]
        fn malloc(size: u32) -> *mut std::ffi::c_void;

        #[cfg(target_pointer_width="32")]
        fn malloc(size: u16) -> *mut std::ffi::c_void;
    }

    unsafe {
        let _p1 = malloc(42); //~ ERROR Undefined Behavior: scalar size mismatch
    };
}
