#![allow(dropping_references)]

fn main() {
    let target = &mut 42;
    let target2 = target as *mut _;
    drop(&mut *target); // reborrow
    // Now make sure our ref is still the only one.
    unsafe { *target2 = 13 }; //~ ERROR: /write access .* tag does not exist in the borrow stack/
    let _val = *target;
}
