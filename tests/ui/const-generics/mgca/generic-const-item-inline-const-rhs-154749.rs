//! Regression test for <https://github.com/rust-lang/rust/issues/154749>.
//@ run-pass
#![feature(gca_min_const_items, generic_const_items)]

const FREE1<T>: usize = const { std::mem::size_of::<*mut T>() };

fn main() {
    assert_eq!(FREE1::<u8>, std::mem::size_of::<*mut u8>());
}
