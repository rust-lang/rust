//! Tests basic pointer coercions

//@ run-pass

pub fn main() {
    // &mut -> &
    let x: &mut isize = &mut 42;
    let _x: &isize = x;
    let _x: &isize = &mut 42;

    // & -> *const
    let x: &isize = &42;
    let _x: *const isize = x;
    let _x: *const isize = &42;

    // &mut -> *const
    let x: &mut isize = &mut 42;
    let _x: *const isize = x;
    let _x: *const isize = &mut 42;

    // *mut -> *const
    let _x: *mut isize = &mut 42;
    let _x: *const isize = x;
}
