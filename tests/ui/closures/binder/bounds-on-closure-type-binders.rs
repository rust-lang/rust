//@ check-fail

#![allow(incomplete_features)]
#![feature(non_lifetime_binders)]
#![feature(closure_lifetime_binder)]

trait Trait {}

fn main() {
    // Regression test for issue #119067
    let _ = for<T: Trait> || -> () {};
    //~^ ERROR bounds cannot be used in this context
    //~| ERROR late-bound type parameter not allowed on closures
}
