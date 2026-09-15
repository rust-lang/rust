//! Regression test for: https://github.com/rust-lang/rust/issues/142722.

#![feature(trivial_bounds)]
#![feature(generic_const_exprs)]
#![feature(min_generic_const_args)]
#![feature(inherent_associated_types)]

struct Foo;
impl Foo {
    const ASSOC_C: usize = todo!();
    //~^ ERROR evaluation panicked: not yet implemented
    fn foo()
    where
        [u8; Self::ASSOC_C]:,
    {
    }
}

pub fn main() {}
