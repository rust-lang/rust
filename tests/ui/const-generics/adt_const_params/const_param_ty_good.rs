//@ check-pass

#![feature(adt_const_params, unsized_const_params)]
#![allow(incomplete_features)]

use std::marker::{ConstParamTy, ConstParamTy_};

#[derive(PartialEq, Eq)]
struct S<T> {
    field: u8,
    gen: T,
}

impl<T: ConstParamTy_> ConstParamTy_ for S<T> {}

#[derive(PartialEq, Eq, ConstParamTy)]
struct D<T> {
    field: u8,
    gen: T,
}

fn check<T: ConstParamTy_ + ?Sized>() {}

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
