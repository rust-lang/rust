// Demonstrates and records a theoretical regressions / breaking changes caused by the
// introduction of async trait bounds.

// Setting the edition to >2015 since we didn't regress `demo! { dyn async }` in Rust 2015.
//@ edition:2018

macro_rules! demo {
    ($ty:ty) => { compile_error!("ty"); }; // KEEP THIS RULE FIRST AND AS IS!
    //~^ ERROR ty
    //~| ERROR ty

    // DON'T MODIFY THE MATCHERS BELOW UNLESS THE ASYNC TRAIT MODIFIER SYNTAX CHANGES!

    (impl $c:ident Trait) => { /* KEEP THIS EMPTY! */ };
    (dyn $c:ident Trait) => { /* KEEP THIS EMPTY! */ };
}

demo! { impl async Trait } //~ ERROR `async` trait bounds are unstable

demo! { dyn async Trait } //~ ERROR `async` trait bounds are unstable

fn main() {}
