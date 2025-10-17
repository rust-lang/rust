// Regression test for issue #118040.
// Ensure that we support assoc const eq bounds where the assoc const comes from a supertrait.

//@ check-pass

#![feature(associated_const_equality, min_generic_const_args)]
#![allow(incomplete_features)]

trait Trait: SuperTrait {}
trait SuperTrait: SuperSuperTrait<i32> {}
trait SuperSuperTrait<T> {
    #[type_const]
    const K: T;
}

fn take(_: impl Trait<K = 0>) {}

fn main() {}
