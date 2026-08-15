//@only-target: linux # `memrchr` is a GNU extension

use std::ptr;

// null is explicitly called out as UB in the C docs for `memchr`.
fn main() {
    unsafe {
        libc::memrchr(ptr::null(), 0, 0); //~ERROR: null pointer
    }
}
