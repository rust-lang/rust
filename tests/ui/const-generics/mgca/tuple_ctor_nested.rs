//@ run-pass
#![feature(min_generic_const_args, adt_const_params)]
#![expect(incomplete_features)]

use std::marker::ConstParamTy;

#[derive(Debug, Eq, PartialEq, ConstParamTy)]
struct Inner(u32);

#[derive(Debug, Eq, PartialEq, ConstParamTy)]
struct Outer(Inner);

#[derive(Debug, Eq, PartialEq, ConstParamTy)]
enum Container<T> {
    Wrap(T),
}

fn with_outer<const O: Outer>() -> Outer {
    O
}

fn with_container<const C: Container<Inner>>() -> Container<Inner> {
    C
}

fn test<const N: u32>() {
    with_outer::<{ Outer(Inner(N)) }>();
    with_outer::<{ Outer(Inner(const { 42 })) }>();

    with_container::<{ Container::Wrap::<Inner>(Inner(N)) }>();
}

fn main() {
    test::<5>();

    let o = with_outer::<{ Outer(Inner(const { 10 })) }>();
    assert_eq!(o, Outer(Inner(10)));
}
