// run-pass
#![feature(const_generics)]
#![allow(incomplete_features)]

fn foo<const N: usize>(v: &[u8; N]) -> &[u8] {
    v
}

fn main() {
    assert_eq!(foo(&[1, 2]), &[1, 2]);
}
