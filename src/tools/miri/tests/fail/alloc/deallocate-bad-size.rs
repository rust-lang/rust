use std::alloc::{Layout, alloc, dealloc};

fn main() {
    unsafe {
        let x = alloc(Layout::from_size_align_unchecked(1, 1));
        dealloc(x, Layout::from_size_align_unchecked(2, 1)); //~ERROR: has size 1 and alignment 1, but gave size 2 and alignment 1
    }
}
