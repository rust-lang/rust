//@ run-pass
#![feature(min_generic_const_args, adt_const_params)]
#![expect(incomplete_features)]
#![allow(dead_code)]

use std::marker::ConstParamTy;

#[derive(Debug, Eq, PartialEq, ConstParamTy)]
struct Point(u32, u32);

#[derive(Debug, Eq, PartialEq, ConstParamTy)]
enum MyEnum<T> {
    Variant(T),
    Other,
}

trait Trait {
    #[type_const]
    const ASSOC: u32;
}

fn with_point<const P: Point>() -> Point {
    P
}

fn with_enum<const E: MyEnum<u32>>() -> MyEnum<u32> {
    E
}

fn test<T: Trait, const N: u32>() {
    with_point::<{ Point(<T as Trait>::ASSOC, N) }>();
}

fn test_basic<const N: u32>() {
    with_point::<{ Point(N, N) }>();
    with_point::<{ Point(const { 5 }, const { 10 }) }>();

    with_enum::<{ MyEnum::Variant::<u32>(N) }>();
    with_enum::<{ MyEnum::Variant::<u32>(const { 42 }) }>();

    with_enum::<{ <MyEnum<u32>>::Variant(N) }>();
}

fn main() {
    test_basic::<5>();

    let p = with_point::<{ Point(const { 1 }, const { 2 }) }>();
    assert_eq!(p, Point(1, 2));

    let e = with_enum::<{ MyEnum::Variant::<u32>(const { 10 }) }>();
    assert_eq!(e, MyEnum::Variant(10));
}
