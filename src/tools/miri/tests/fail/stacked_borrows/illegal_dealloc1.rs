use std::alloc::{Layout, alloc, dealloc};

fn main() {
    unsafe {
        let x = alloc(Layout::from_size_align_unchecked(1, 1));
        let ptr1 = (&mut *x) as *mut u8;
        let ptr2 = (&mut *ptr1) as *mut u8;
        // Invalidate ptr2 by writing to ptr1.
        ptr1.write(0);
        // Deallocate through ptr2.
        dealloc(ptr2, Layout::from_size_align_unchecked(1, 1));
        //~^ERROR: /deallocation .* tag does not exist in the borrow stack/
    }
}
