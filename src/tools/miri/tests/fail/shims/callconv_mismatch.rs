extern "Rust" {
    fn pipe(fds: *mut std::ffi::c_int) -> std::ffi::c_int;
}

// Test the error for calling convention mismatch. 
fn main() {
    let mut fds = [-1, -1];
    let res = unsafe { pipe(fds.as_mut_ptr()) };
    //~^ ERROR: calling a function with calling convention C using caller calling convention Rust
}