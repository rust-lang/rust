//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows

// This makes a ref that was passed to us via &mut alias with things it should not alias with
fn retarget(x: &mut &u32, target: &mut u32) {
    unsafe {
        *x = &mut *(target as *mut _);
    }
}

fn main() {
    let target = &mut 42;
    let mut target_alias = &42; // initial dummy value
    retarget(&mut target_alias, target);
    // now `target_alias` points to the same thing as `target`
    *target = 13;
    let _val = *target_alias;
    //~[stack]^ ERROR: /read access .* tag does not exist in the borrow stack/
    //~[tree]| ERROR: /read access through .* is forbidden/
}
