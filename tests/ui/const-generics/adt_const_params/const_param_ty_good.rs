// check-pass
#![allow(incomplete_features)]
#![feature(adt_const_params)]
use std::marker::ConstParamTy;

#[derive(PartialEq, Eq)]
struct S<T> {
    field: u8,
    gen: T,
}

impl<T: ConstParamTy> ConstParamTy for S<T> {}

#[derive(PartialEq, Eq, ConstParamTy)]
struct D<T> {
    field: u8,
    gen: T,
}


fn check<T: ConstParamTy + ?Sized>() {}

fn main() {
    check::<u8>();
    check::<u16>();
    check::<u32>();
    check::<u64>();
    check::<u128>();

    check::<i8>();
    check::<i16>();
    check::<i32>();
    check::<i64>();
    check::<i128>();

    check::<char>();
    check::<bool>();
    check::<str>();

    check::<&u8>();
    check::<&str>();
    check::<[usize]>();
    check::<[u16; 0]>();
    check::<[u8; 42]>();

    check::<S<u8>>();
    check::<S<[&[bool]; 8]>>();

    check::<D<u8>>();
    check::<D<[&[bool]; 8]>>();

    check::<()>();
    check::<(i32,)>();
    check::<(D<u8>, D<i32>)>();
}
