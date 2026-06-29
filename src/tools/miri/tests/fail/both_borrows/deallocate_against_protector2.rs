//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows

// Ensure that even a ZST prevents the reference from being used for deallocation.
// The `nofree` attributes we add in LLVM IR rely on this.

use std::alloc::Layout;

fn inner(x: &mut (), f: fn(&mut ())) {
    // `f` may mutate, but it may not deallocate!
    f(x)
}

fn main() {
    let ptr = Box::leak(Box::new(0i32)) as *mut i32;
    inner(unsafe { &mut *(ptr as *mut ()) }, |x| unsafe {
        let raw = x as *mut _ as *mut i32;
        // Avoid ever creating a `Box`, we don't want any implicit accesses.
        std::alloc::dealloc(raw.cast(), Layout::new::<i32>());
        //~[tree]^ERROR: /deallocation through .* is forbidden/
        //~[stack]|ERROR: tag does not exist in the borrow stack for this location
    });
}
