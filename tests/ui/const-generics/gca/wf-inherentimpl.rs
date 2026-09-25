//@[next] check-pass
//@ revisions: next old
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
#![feature(inherent_associated_types)]
#![feature(gca_const_items, gca_min_const_items)]
//[old]~^ ERROR `gca_const_items` requires -Znext-solver=globally to be enabled
use std::gca;
struct Foo<const A: usize>;
impl<const A: usize> Foo<A> {
    const SIZE: usize = { todo!() };
    fn to_bytes() -> [u8; gca!(Self::SIZE)] {
        todo!()
    }
}
fn main() {}
