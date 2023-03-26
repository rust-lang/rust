//@compile-flags: -Zmiri-tree-borrows -Zmiri-tag-gc=0

#[path = "../../utils/mod.rs"]
mod utils;
use utils::macros::*;

// To check that a reborrow is counted as a Read access, we use a reborrow
// with no additional Read to Freeze an Active pointer.

fn main() {
    unsafe {
        let parent = &mut 0u8;
        name!(parent);
        let alloc_id = alloc_id!(parent);
        let x = &mut *parent;
        name!(x);
        *x = 0; // x is now Active
        print_state!(alloc_id);
        let y = &mut *parent;
        name!(y);
        // Check in the debug output that x has been Frozen by the reborrow
        print_state!(alloc_id);
    }
}
