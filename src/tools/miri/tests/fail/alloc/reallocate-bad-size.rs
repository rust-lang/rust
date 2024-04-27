use std::alloc::{alloc, realloc, Layout};

//@error-in-other-file: has size 1 and alignment 1, but gave size 2 and alignment 1

fn main() {
    unsafe {
        let x = alloc(Layout::from_size_align_unchecked(1, 1));
        let _y = realloc(x, Layout::from_size_align_unchecked(2, 1), 1);
    }
}
