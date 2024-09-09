//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
//@[tree]error-in-other-file: /deallocation .* is forbidden/
use std::alloc::{alloc, dealloc, Layout};

// `x` is strongly protected but covers zero bytes.
// Let's see if deallocating the allocation x points to is UB:
// in TB, it is UB, but in SB it is not.
fn test(_x: &mut (), ptr: *mut u8, l: Layout) {
    unsafe { dealloc(ptr, l) };
}

fn main() {
    let l = Layout::from_size_align(1, 1).unwrap();
    let ptr = unsafe { alloc(l) };
    unsafe { test(&mut *ptr.cast::<()>(), ptr, l) };
    // In SB the test would pass if it weren't for this line.
    unsafe { std::hint::unreachable_unchecked() }; //~[stack] ERROR: unreachable
}
