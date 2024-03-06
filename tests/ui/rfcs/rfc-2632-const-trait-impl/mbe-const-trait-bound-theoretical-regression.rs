// Demonstrates and records a theoretical regressions / breaking changes caused by the
// introduction of const trait bounds.

// Setting the edition to 2018 since we don't regress `demo! { dyn const }` in Rust <2018.
//@ edition:2018

macro_rules! demo {
    ($ty:ty) => { compile_error!("ty"); };
    //~^ ERROR ty
    //~| ERROR ty
    (impl $c:ident Trait) => {};
    (dyn $c:ident Trait) => {};
}

demo! { impl const Trait }
//~^ ERROR const trait impls are experimental

demo! { dyn const Trait }
//~^ ERROR const trait impls are experimental

fn main() {}
