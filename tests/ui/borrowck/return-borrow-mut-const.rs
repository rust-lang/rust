// See PR #104857 for details

// don't want to tie this test to the lint, even though it's related
#![allow(const_item_mutation)]

fn main() {}

const X: i32 = 42;

fn borrow_const_immut() -> &'static i32 {
    &X
}

fn borrow_const_immut_explicit_return() -> &'static i32 {
    return &X;
}

fn borrow_const_immut_into_temp() -> &'static i32 {
    let x_ref = &X;
    x_ref
}

fn borrow_const_mut() -> &'static mut i32 {
    return &mut X; //~ ERROR
}

fn borrow_const_mut_explicit_return() -> &'static mut i32 {
    return &mut X; //~ ERROR
}

fn borrow_const_mut_into_temp() -> &'static mut i32 {
    let x_ref = &mut X;
    x_ref //~ ERROR
}
