//@ check-pass
// FIXME(mgca): This should error

#![feature(min_generic_const_args, adt_const_params)]
#![expect(incomplete_features)]

#[derive(Eq, PartialEq, std::marker::ConstParamTy)]
struct Foo<T> { field: T }

fn accepts<const N: Foo<u8>>() {}

fn bar<const N: bool>() {
    // `N` is not of type `u8` but we don't actually check this anywhere yet
    accepts::<{ Foo::<u8> { field: N }}>();
}

fn main() {}
