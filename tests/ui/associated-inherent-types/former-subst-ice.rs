//@ check-pass

#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

struct Cont<T>(T);

impl<T: Copy> Cont<T> {
    type Out = Vec<T>;
}

pub fn weird<T: Copy>(x: T) {
    let _: Cont<_>::Out = vec![true];
}

fn main() {}
