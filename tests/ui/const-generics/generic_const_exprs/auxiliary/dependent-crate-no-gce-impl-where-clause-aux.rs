#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub trait Conf {
    const LAYER_COUNT: usize;
}

pub trait Buggy {
    const TEST: u8 = 0;
}

impl<T: Conf> Buggy for T where [(); Self::LAYER_COUNT]: {}
