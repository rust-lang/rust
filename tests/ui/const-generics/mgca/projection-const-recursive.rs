//! See also <https://github.com/rust-lang/rust/issues/153831>
//@ check-fail
//@compile-flags: -Znext-solver=globally --emit=obj
#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

trait Trait {
    type const A: ();
}

impl Trait for () {
    type const A: () = <() as Trait>::A;
    //~^ ERROR: overflow evaluating the requirement `<() as Trait>::A == _`
    //~| ERROR: overflow evaluating the requirement `<() as Trait>::A == _`
    //~| ERROR: type annotations needed
}

fn main() {
    <() as Trait>::A;
}
