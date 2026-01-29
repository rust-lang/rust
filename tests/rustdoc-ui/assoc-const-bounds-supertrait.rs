// Regression test for issue #151511.
// Ensure that rustdoc doesn't ICE when encountering assoc const eq bounds
// where the assoc const comes from a supertrait.

#![feature(
    min_generic_const_args,
    adt_const_params,
    unsized_const_params,
    generic_const_parameter_types
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
//~^ ERROR anonymous constants referencing generics are not yet supported

fn main() {}
