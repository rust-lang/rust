// Regression test for issue #118040.
// Ensure that we support assoc const eq bounds where the assoc const comes from a supertrait.

//@ check-pass

#![feature(associated_const_equality)]

trait Trait: SuperTrait {}
trait SuperTrait: SuperSuperTrait<i32> {}
trait SuperSuperTrait<T> {
    const K: T;
}

fn take(_: impl Trait<K = 0>) {}

fn main() {}
