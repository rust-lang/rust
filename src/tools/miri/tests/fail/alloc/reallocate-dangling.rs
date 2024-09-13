use std::alloc::{alloc, dealloc, realloc, Layout};

fn main() {
    unsafe {
        let x = alloc(Layout::from_size_align_unchecked(1, 1));
        dealloc(x, Layout::from_size_align_unchecked(1, 1));
        let _z = realloc(x, Layout::from_size_align_unchecked(1, 1), 1); //~ERROR: has been freed
    }
}
