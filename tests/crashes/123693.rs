//@ known-bug: #123693

#![feature(transmutability)]

mod assert {
    use std::mem::{Assume, TransmuteFrom};

    pub fn is_transmutable<Src, Dst>()
    where
        Dst: TransmuteFrom<Src, { Assume::NOTHING }>,
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
