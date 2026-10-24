//! Ensure that we do not permit mutating the part of an interior mutable static
//! that is outside the `Cell`.

//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows

use std::sync::atomic::*;

static X: (i32, AtomicI32) = (0, AtomicI32::new(1));

fn main() {
    let ptr = &raw const X;
    unsafe { ptr.cast_mut().write((1, AtomicI32::new(0))) };
    //~[stack]^ERROR: that tag only grants SharedReadOnly permission
    //~[tree]|ERROR: /write access .* is forbidden/
}
