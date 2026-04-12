// FIXME: should this pass?
//@ check-pass
#![feature(align_type)]

use std::marker::PhantomData;
use std::mem::Align;

struct TypeFromTheTernaryDimension {
    align: Align<3>,
}

struct InsidePhantom {
    phantom: PhantomData<Align<9999>>,
}

const _: () = {
    assert!(size_of::<InsidePhantom>() == 0);
    assert!(align_of::<InsidePhantom>() == 1);
};

fn main() {}
