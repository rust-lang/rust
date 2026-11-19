//@ compile-flags: -Znext-solver=globally

#![feature(generic_const_items, gca_const_items, gca_macroless_args, gca_min_const_items)]

pub const P<const N: usize>: usize = N;

pub struct S;

impl S {
    pub fn f<const N: usize>(_: [u8; P::<N>]) {}

    pub const C<const N: usize>: [u8; P::<N>] = [0; P::<N>];
}

pub trait Tr {
    type T<const N: usize>;
}

impl Tr for S {
    type T<const N: usize> = [u8; P::<N>];
}
