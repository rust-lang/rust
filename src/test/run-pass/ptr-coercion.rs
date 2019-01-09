#![allow(unused_variables)]
// Test coercions between pointers which don't do anything fancy like unsizing.

// pretty-expanded FIXME #23616

pub fn main() {
    // &mut -> &
    let x: &mut isize = &mut 42;
    let x: &isize = x;

    let x: &isize = &mut 42;

    // & -> *const
    let x: &isize = &42;
    let x: *const isize = x;

    let x: *const isize = &42;

    // &mut -> *const
    let x: &mut isize = &mut 42;
    let x: *const isize = x;

    let x: *const isize = &mut 42;

    // *mut -> *const
    let x: *mut isize = &mut 42;
    let x: *const isize = x;
}
