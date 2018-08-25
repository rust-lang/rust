// Validation makes this fail in the wrong place
// compile-flags: -Zmir-emit-validate=0

// error-pattern: invalid enum discriminant

use std::mem;

#[repr(C)]
pub enum Foo {
    A, B, C, D
}

fn main() {
    let f = unsafe { std::mem::transmute::<i32, Foo>(42) };
    let _ = mem::discriminant(&f);
}
