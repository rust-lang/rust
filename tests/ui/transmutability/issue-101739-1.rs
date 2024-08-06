#![feature(transmutability)]

mod assert {
    use std::mem::BikeshedIntrinsicFrom;

    pub fn is_transmutable<Src, const ASSUME_ALIGNMENT: bool>()
    where
        Dst: BikeshedIntrinsicFrom<Src, ASSUME_ALIGNMENT>, //~ ERROR cannot find type `Dst` in this scope
        //~^ the constant `ASSUME_ALIGNMENT` is not of type `Assume`
        //~| ERROR: mismatched types
    {
    }
}

fn via_const() {
    struct Src;

    assert::is_transmutable::<Src, false>();
}

fn main() {}
