// Regression test for <github.com/rust-lang/rust/issues/145185>.

mod module {
    pub trait Trait {
        fn method(&self);
    }
}

// Note that we do not import Trait
use std::ops::Deref;

fn foo(x: impl Deref<Target: module::Trait>) {
    x.method();
    //~^ ERROR no method named `method` found for type parameter
}

fn main() {}
