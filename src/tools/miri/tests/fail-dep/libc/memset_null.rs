use std::ptr;

// null is explicitly called out as UB in the C docs for `memset`.
fn main() {
    unsafe {
        libc::memset(ptr::null_mut(), 0, 0); //~ERROR: null pointer
    }
}
