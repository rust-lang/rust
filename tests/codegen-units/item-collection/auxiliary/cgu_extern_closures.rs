//@ compile-flags: -Zinline-mir=no

#![crate_type = "lib"]

#[inline]
pub fn inlined_fn(x: i32, y: i32) -> i32 {
    let closure = |a, b| a + b;

    closure(x, y)
}

pub fn inlined_fn_generic<T>(x: i32, y: i32, z: T) -> (i32, T) {
    let closure = |a, b| a + b;

    (closure(x, y), z)
}

pub fn non_inlined_fn(x: i32, y: i32) -> i32 {
    let closure = |a, b| a + b;

    closure(x, y)
}
