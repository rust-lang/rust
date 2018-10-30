// The compiler inserts some reborrows, enable optimizations to
// get rid of them.
// compile-flags: -Zmir-opt-level=1

use std::mem;

// This is an example of a piece of code that intuitively seems like we might
// want to reject it, but that doesn't turn out to be possible.

fn main() {
    let target = 42;
    // Make sure a cannot use a raw-tagged `&mut` pointing to a frozen location, not
    // even to create a raw.
    let reference = &target; // freeze
    let ptr = reference as *const _ as *mut i32; // raw ptr, with raw tag
    let mut_ref: &mut i32 = unsafe { mem::transmute(ptr) }; // &mut, with raw tag
    // Now we have an &mut to a frozen location, but that is completely normal:
    // We'd just unfreeze the location if we used it.
    let bad_ptr = mut_ref as *mut i32; // even just creating this is like a use of `mut_ref`.
    // That violates the location being frozen!  However, we do not properly detect this:
    // We first see a `&mut` with a `Raw` tag being deref'd for a frozen location,
    // which can happen legitimately if the compiler optimized away an `&mut*` that
    // turns a raw into a `&mut`.  Next, we create a raw ref to a frozen location
    // from a `Raw` tag, which can happen legitimately when interior mutability
    // is involved.
    let _val = *reference; // Make sure it is still frozen.

    // We only actually unfreeze once we muteate through the bad pointer.
    unsafe { *bad_ptr = 42 }; //~ ERROR does not exist on the stack
    let _val = *reference;
}
