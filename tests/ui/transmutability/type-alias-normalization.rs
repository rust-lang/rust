//! regression test for https://github.com/rust-lang/rust/issues/151462
//@compile-flags: -Znext-solver=globally
#![feature(lazy_type_alias, transmutability)]
#![allow(incomplete_features)]
mod assert {
    use std::mem::{Assume, TransmuteFrom};

    pub fn is_maybe_transmutable<Src, Dst>()
    where
        Src: TransmuteFrom<
            Src,
            {
                Assume {
                    alignment: true,
                    lifetimes: true,
                    safety: true,
                    validity: true,
                }
            },
        >,
    {
    }
}

fn test() {
    type JustUnit = ();
    assert::is_maybe_transmutable::<JustUnit, ()>();
    //~^ ERROR `JustUnit` cannot be safely transmuted into `JustUnit`
}

fn main() {}
