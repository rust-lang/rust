// build-pass

#![feature(adt_const_params)]
#![allow(incomplete_features)]

use std::marker::ConstParamTy;

#[derive(PartialEq, Eq, ConstParamTy)]
pub struct UnitDims {
    pub time: u8,
    pub length: u8,
}

pub struct UnitValue<const DIMS: UnitDims>;

impl<const DIMS: UnitDims> UnitValue<DIMS> {
    fn crash() {}
}

fn main() {
    UnitValue::<{ UnitDims { time: 1, length: 2 } }>::crash();
}
