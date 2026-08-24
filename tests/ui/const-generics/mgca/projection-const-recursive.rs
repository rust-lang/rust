//@ check-fail
//@ compile-flags: -Znext-solver=globally

// Regression test for <https://github.com/rust-lang/rust/issues/153831>

#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

trait Trait {
    type const A: ();
}

impl Trait for () {
    type const A: () = <() as Trait>::A;
    //~^ ERROR: overflow evaluating the requirement `<() as Trait>::A == _`
    //~| ERROR: overflow evaluating the requirement `the constant `<() as Trait>::A` has type `()``
}

fn main() {
    <() as Trait>::A;
}
