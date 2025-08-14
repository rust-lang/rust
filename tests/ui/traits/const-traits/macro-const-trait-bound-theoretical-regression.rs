// Demonstrates and records a theoretical regressions / breaking changes caused by the
// introduction of const trait bounds.

// Setting the edition to 2018 since we don't regress `demo! { dyn const }` in Rust <2018.
//@ edition:2018

trait Trait {}

macro_rules! demo {
    (impl $c:ident Trait) => { impl $c Trait {} };
    //~^ ERROR inherent
    //~| WARN trait objects without an explicit `dyn` are deprecated
    //~| WARN this is accepted in the current edition
    (dyn $c:ident Trait) => { dyn $c Trait {} }; //~ ERROR macro expansion
}

demo! { impl const Trait }
//~^ ERROR const trait impls are experimental

demo! { dyn const Trait }

fn main() {}
