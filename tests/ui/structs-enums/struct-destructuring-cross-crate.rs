//@ run-pass
//@ aux-build:struct_destructuring_cross_crate.rs


extern crate struct_destructuring_cross_crate;

pub fn main() {
    let x = struct_destructuring_cross_crate::S { x: 1, y: 2 };
    let struct_destructuring_cross_crate::S { x: a, y: b } = x;
    assert_eq!(a, 1);
    assert_eq!(b, 2);
}
