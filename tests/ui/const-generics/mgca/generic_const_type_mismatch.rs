//! Regression test for <https://github.com/rust-lang/rust/issues/150983>
#![feature(
    adt_const_params,
    const_param_ty_trait,
    gca_adts,
    gca_min_const_items,
    generic_const_items,
    generic_const_parameter_types
)]

use std::gca;
use std::marker::ConstParamTy_;

struct Foo<T> {
    field: T,
}

const WRAP<T: ConstParamTy_>: T = gca!(Foo::<T> { field: 1 });
//~^ ERROR: type annotations needed for the literal

fn main() {}
