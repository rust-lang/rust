#![allow(unused_variables)]

fn main() {
    let target = &mut 42;
    let target2 = target as *mut _;
    drop(&mut *target); // reborrow
    // Now make sure our ref is still the only one
    unsafe { *target2 = 13; } // invalidate our ref
    let _val = *target; //~ ERROR does not exist on the stack
}
