// Demonstrates and records a theoretical regressions / breaking changes caused by the
// introduction of async trait bounds.

// Setting the edition to 2018 since we don't regress `demo! { dyn async }` in Rust <2018.
//@ edition:2018

macro_rules! demo {
    ($ty:ty) => { compile_error!("ty"); };
    //~^ ERROR ty
    //~| ERROR ty
    (impl $c:ident Trait) => {};
    (dyn $c:ident Trait) => {};
}

demo! { impl async Trait }
//~^ ERROR async closures are unstable

demo! { dyn async Trait }
//~^ ERROR async closures are unstable

fn main() {}
