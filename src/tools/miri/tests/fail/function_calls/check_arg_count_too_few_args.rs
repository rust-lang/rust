fn main() {
    extern "C" {
        fn malloc() -> *mut std::ffi::c_void;
    }

    unsafe {
        let _ = malloc(); //~ ERROR: Undefined Behavior: incorrect number of arguments for `malloc`: got 0, expected 1
    };
}
