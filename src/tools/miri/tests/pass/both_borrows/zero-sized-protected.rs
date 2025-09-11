//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
use std::alloc::{Layout, alloc, dealloc};

// `x` is strongly protected but covers zero bytes.
// This should never be UB.
fn test(_x: &mut (), ptr: *mut u8, l: Layout) {
    unsafe { dealloc(ptr, l) };
}

fn main() {
    let l = Layout::from_size_align(1, 1).unwrap();
    let ptr = unsafe { alloc(l) };
    unsafe { test(&mut *ptr.cast::<()>(), ptr, l) };
}
