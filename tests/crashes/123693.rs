//@ known-bug: #123693

#![feature(transmutability)]

mod assert {
    use std::mem::{Assume, BikeshedIntrinsicFrom};

    pub fn is_transmutable<Src, Dst>()
    where
        Dst: BikeshedIntrinsicFrom<Src, { Assume::NOTHING }>,
    {
    }
}

enum Lopsided {
    Smol(()),
    Lorg(bool),
}

fn should_pad_variants() {
    assert::is_transmutable::<Lopsided, ()>();
}
