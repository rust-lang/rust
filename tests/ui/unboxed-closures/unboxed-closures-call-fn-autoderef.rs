// run-pass
#![allow(unused_imports)]
// Test that the call operator autoderefs when calling a bounded type parameter.

use std::ops::FnMut;

fn call_with_2(x: &fn(isize) -> isize) -> isize
{
    x(2) // look ma, no `*`
}

fn subtract_22(x: isize) -> isize { x - 22 }

pub fn main() {
    let subtract_22: fn(isize) -> isize = subtract_22;
    let z = call_with_2(&subtract_22);
    assert_eq!(z, -20);
}
