//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows

// This used to accidentally be accepted by SB.
fn main() {
    unsafe {
        let mut root = 0;
        let to = &mut root as *mut i32;
        *to = 0;
        let _val = root;
        *to = 0;
        //~[stack]^ ERROR: tag does not exist in the borrow stack
        //~[tree]| ERROR: forbidden
    }
}
