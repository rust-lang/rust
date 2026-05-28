//@ run-pass
#![feature(min_generic_const_args, adt_const_params)]
#![expect(incomplete_features)]

use std::marker::ConstParamTy;

#[derive(Debug, Eq, PartialEq, ConstParamTy)]
enum Option<T> {
    Some(T),
}

fn with_option<const O: Option<u32>>() -> Option<u32> {
    O
}

fn test<const N: u32>() {
    with_option::<{ <Option<u32>>::Some(N) }>();
    with_option::<{ <Option<u32>>::Some(const { 42 }) }>();
}

fn main() {
    test::<5>();

    let o = with_option::<{ <Option<u32>>::Some(const { 10 }) }>();
    assert_eq!(o, Option::Some(10));
}
