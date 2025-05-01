//@ignore-target: windows # No libc pipe on Windows

// Declare a non-variadic function as variadic.
extern "C" {
    fn pipe(fds: *mut std::ffi::c_int, ...) -> std::ffi::c_int;
}

// Test the error caused by invoking non-vararg shim with a vararg import.
fn main() {
    let mut fds = [-1, -1];
    let res = unsafe { pipe(fds.as_mut_ptr()) };
    //~^ ERROR: ABI mismatch: calling a non-variadic function with a variadic caller-side signature
    assert_eq!(res, 0);
}
