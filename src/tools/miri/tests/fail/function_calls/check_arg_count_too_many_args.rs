fn main() {
    extern "C" {
        fn malloc(_: i32, _: i32) -> *mut std::ffi::c_void;
    }

    unsafe {
        let _ = malloc(1, 2); //~ ERROR: Undefined Behavior: incorrect number of arguments for `malloc`: got 2, expected 1
    };
}
