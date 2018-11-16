// We fail to detect this when neither this nor libstd are optimized/have retagging.
// FIXME: Investigate that.
// compile-flags: -Zmir-opt-level=0

#![allow(unused_variables)]

fn main() {
    let target = &mut 42;
    let target2 = target as *mut _;
    drop(&mut *target); // reborrow
    // Now make sure our ref is still the only one.
    unsafe { *target2 = 13; } //~ ERROR does not exist on the stack
    let _val = *target;
}
