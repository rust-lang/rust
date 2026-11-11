// This once used to ICE with "const parameter `P/#1` (P/#1/1) out of range when
// instantiating args=[?2c]".
//
// issue: <https://github.com/rust-lang/rust/issues/145824>
//
//@ check-pass
//@ compile-flags: -Znext-solver=globally

#![feature(gca_min_const_items, gca_const_items)]

trait Trait {
    const N: usize;
}

impl<const M: usize> Trait for [(); M] {
    const N: usize = M;
}

fn empty<const O: usize, const P: usize>() -> [(); P] {
    new()
}

fn new<T, const Q: usize>() -> T
where
    T: Trait<N = { Q }>,
{
    loop {}
}

fn main() {}
