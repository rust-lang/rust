//@ignore-target-windows: No libc on Windows
//@ignore-target-apple: No `memrchr` on some apple targets

use std::ptr;

// null is explicitly called out as UB in the C docs.
fn main() {
    unsafe {
        libc::memrchr(ptr::null(), 0, 0); //~ERROR: dangling
    }
}
