// Regression test for issue #118040.
// Ensure that we support assoc const eq bounds where the assoc const comes from a supertrait.

//@ check-pass

#![feature(
    min_generic_const_args,
    adt_const_params,
    unsized_const_params,
    generic_const_parameter_types,
)]
#![allow(incomplete_features)]

use std::marker::ConstParamTy_;

trait Trait: SuperTrait {}
trait SuperTrait: SuperSuperTrait<i32> {}
trait SuperSuperTrait<T: ConstParamTy_> {
    #[type_const]
    const K: T;
}

fn take(_: impl Trait<K = 0>) {}

fn main() {}
