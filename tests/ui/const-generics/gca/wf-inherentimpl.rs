//@[next] check-pass
//@ revisions: next old
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
#![feature(inherent_associated_types)]
#![feature(macroless_generic_const_args)]
#![feature(generic_const_args, min_generic_const_args)]
//[old]~^ ERROR `generic_const_args` requires -Znext-solver=globally to be enabled
struct Foo<const A: usize>;
impl<const A: usize> Foo<A> {
    const SIZE: usize = { todo!() };
    fn to_bytes() -> [u8; Self::SIZE] {
        todo!()
    }
}
fn main() {}
