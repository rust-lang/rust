// Validation makes this fail in the wrong place
// compile-flags: -Zmiri-disable-validation

// error-pattern: invalid enum discriminant

use std::mem;

#[repr(C)]
pub enum Foo {
    A, B, C, D
}

fn main() {
    let f = unsafe { std::mem::transmute::<i32, Foo>(42) };
    let _val = mem::discriminant(&f);
}
