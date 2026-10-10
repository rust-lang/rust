//@ known-bug: #149748
//@ edition: 2024
//@ compile-flags: -Zmir-enable-passes=+Inline -Zmir-enable-passes=+ReferencePropagation -Zlint-mir

#![feature(gen_blocks)]
gen fn foo(z: i32) -> i32 {
    yield z;
    z;
}
pub fn main() {
    let mut iter = foo(3);
    assert_eq!(iter.next(), Some(3))
}
