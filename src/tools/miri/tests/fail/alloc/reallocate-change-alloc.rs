use std::alloc::{alloc, realloc, Layout};

fn main() {
    unsafe {
        let x = alloc(Layout::from_size_align_unchecked(1, 1));
        let _y = realloc(x, Layout::from_size_align_unchecked(1, 1), 1);
        let _z = *x; //~ ERROR: dereferenced after this allocation got freed
    }
}
