// Demonstrates and records a theoretical regressions / breaking changes caused by the
// introduction of `const` and `[const]` trait bounds.

// Setting the edition to >2015 since we didn't regress `demo! { dyn const }` in Rust 2015.
// See also test `traits/const-traits/macro-dyn-const-2015.rs`.
//@ edition:2018

trait Trait {}

macro_rules! demo {
    ($ty:ty) => { compile_error!("ty"); }; // KEEP THIS RULE FIRST AND AS IS!
    //~^ ERROR ty
    //~| ERROR ty
    //~| ERROR ty
    //~| ERROR ty

    // DON'T MODIFY THE MATCHERS BELOW UNLESS THE CONST TRAIT MODIFIER SYNTAX CHANGES!

    (impl $c:ident Trait) => { /* KEEP THIS EMPTY! */ };
    (dyn $c:ident Trait) => { /* KEEP THIS EMPTY! */ };

    (impl [const] Trait) => { /* KEEP THIS EMPTY! */ };
    (dyn [const] Trait) => { /* KEEP THIS EMPTY! */ };
}

demo! { impl const Trait } //~ ERROR const trait impls are experimental
demo! { dyn const Trait } //~ ERROR const trait impls are experimental

demo! { impl [const] Trait } //~ ERROR const trait impls are experimental
demo! { dyn [const] Trait } //~ ERROR const trait impls are experimental

fn main() {}
