//@no-rustfix
#![allow(unused)]
#![warn(clippy::as_ptr_cast_mut)]

fn main() {
    let mut string = String::new();

    // the `*mut _` is actually necessary since it does two things at once:
    // - changes the mutability (caught by the lint)
    // - changes the type
    //
    // and so replacing this with `as_mut_ptr` removes the second thing,
    // resulting in a type mismatch
    let _: *mut i8 = string.as_ptr() as *mut _;
    //~^ as_ptr_cast_mut
}
