fn main() {
    extern "C" {
        fn malloc() -> *mut std::ffi::c_void;
    }

    unsafe {
        let _ = malloc(); //~ ERROR: Undefined Behavior: incorrect number of arguments: got 0, expected 1
    };
}
