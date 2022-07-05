use std::alloc::{alloc, dealloc, Layout};

//@error-pattern: dereferenced after this allocation got freed

fn main() {
    unsafe {
        let x = alloc(Layout::from_size_align_unchecked(1, 1));
        dealloc(x, Layout::from_size_align_unchecked(1, 1));
        dealloc(x, Layout::from_size_align_unchecked(1, 1));
    }
}
