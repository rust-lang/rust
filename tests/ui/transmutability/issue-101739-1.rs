#![feature(transmutability)]

mod assert {
    use std::mem::BikeshedIntrinsicFrom;

    pub fn is_transmutable<Src, Context, const ASSUME_ALIGNMENT: bool>()
    where
        Dst: BikeshedIntrinsicFrom<Src, Context, ASSUME_ALIGNMENT>, //~ ERROR cannot find type `Dst` in this scope
        //~^ the constant `ASSUME_ALIGNMENT` is not of type `Assume`
    {
    }
}

fn via_const() {
    struct Context;
    struct Src;

    assert::is_transmutable::<Src, Context, false>();
}

fn main() {}
