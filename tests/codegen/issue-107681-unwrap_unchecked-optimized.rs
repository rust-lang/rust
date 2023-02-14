// compile-flags: -C opt-level=3

#![crate_type = "lib"]

extern crate core;
use core::{iter::Copied, slice::Iter};

// Make sure that the use of unwrap_unchecked is optimized accordingly.
// i.e., there are no branch jumps.
pub unsafe fn unwrap_unchecked_optimized(x: &mut Copied<Iter<'_, u32>>) -> u32 {
    // CHECK-NOT: br
    x.next().unwrap_unchecked()
}
