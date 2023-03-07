// check-pass

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use std::marker::PhantomData;

pub trait Bytes {
    const BYTES: usize;
}

#[derive(Clone, Debug)]
pub struct Conster<OT>
where
    OT: Bytes,
    [(); OT::BYTES]: Sized,
{
    _offset_type: PhantomData<fn(OT) -> OT>,
}

impl<OT> Conster<OT>
where
    OT: Bytes,
    [(); OT::BYTES]: Sized,
{
    pub fn new() -> Self {
        Conster { _offset_type: PhantomData }
    }
}

pub fn make_conster<COT>() -> Conster<COT>
where
    COT: Bytes,
    [(); COT::BYTES]: Sized,
{
    Conster::new()
}

fn main() {}
