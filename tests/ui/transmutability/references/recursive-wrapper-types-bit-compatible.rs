// check-fail
#![feature(transmutability)]

mod assert {
    use std::mem::{Assume, BikeshedIntrinsicFrom};
    pub struct Context;

    pub fn is_maybe_transmutable<Src, Dst>()
    where
        Dst: BikeshedIntrinsicFrom<Src, Context, {
            Assume {
                alignment: true,
                lifetimes: false,
                safety: true,
                validity: false,
            }
        }>
    {}
}

fn main() {
    #[repr(C)] struct A(bool, &'static A);
    #[repr(C)] struct B(u8, &'static B);
    // FIXME(bryangarza): Make 2 variants of this test, depending on mutability.
    // Right now, we are being strict by default and checking A->B and B->A both.
    assert::is_maybe_transmutable::<&'static A, &'static B>(); //~ ERROR `B` cannot be safely transmuted into `A`
}
