#![feature(transmutability)]

mod assert {
    use std::mem::TransmuteFrom;

    pub fn is_transmutable<Src, const ASSUME_ALIGNMENT: bool>()
    where
        Dst: TransmuteFrom<Src, ASSUME_ALIGNMENT>, //~ ERROR cannot find type `Dst` in this scope
    {
    }
}

fn via_const() {
    struct Src;

    assert::is_transmutable::<Src, false>();
}

fn main() {}
