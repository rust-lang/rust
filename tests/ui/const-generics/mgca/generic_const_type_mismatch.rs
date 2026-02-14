//! Regression test for <https://github.com/rust-lang/rust/issues/150983>
#![expect(incomplete_features)]
#![feature(
    adt_const_params,
    generic_const_items,
    generic_const_parameter_types,
    min_generic_const_args,
    unsized_const_params
)]
use std::marker::ConstParamTy_;

struct Foo<T> {
    field: T,
}

type const WRAP<T : ConstParamTy_> : T = Foo::<T>{field : 1};
//~^ ERROR: type annotations needed for the literal

fn main() {}
