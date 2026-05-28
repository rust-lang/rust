//! Regression test for <https://github.com/rust-lang/rust/issues/149287>
//! Ensure that rustdoc does not ICE when a body-less type const is used
//! as an associated const.
//@ check-pass

#![feature(min_generic_const_args)]

pub trait Tr {
    type const SIZE: usize;
}

fn mk_array<T: Tr>() -> [(); <T as Tr>::SIZE] {
    [(); T::SIZE]
}

fn main() {}
