use std::ptr;

// null is explicitly called out as UB in the C docs.
fn main() {
    unsafe {
        libc::memchr(ptr::null(), 0, 0); //~ERROR: null pointer
    }
}
