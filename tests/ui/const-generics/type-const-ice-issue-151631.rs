// issue: <https://github.com/rust-lang/rust/issues/151631>
//@ compile-flags: -Znext-solver
#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

trait SuperTrait {}
trait Trait: SuperTrait {
    type const K: u32;
}
impl Trait for () { //~ ERROR: the trait bound `(): SuperTrait` is not satisfied
    type const K: u32 = const { 1 };
}

fn check(_: impl Trait<K = 0>) {}

fn main() {
    check(()); //~ ERROR: the trait bound `(): SuperTrait` is not satisfied
}
