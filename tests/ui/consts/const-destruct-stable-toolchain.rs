// Test that the suggestion to constrain a type parameter that is dropped in a const
// function with a `[const] Destruct` bound is only offered on nightly, since the bound
// requires an unstable feature.
//
//@ needs-target-std
//@ ignore-backends: gcc
//
//@ revisions: stable nightly
//
//@[stable] act-as-stable
//@[nightly] only-nightly
const fn const_drop<T>(_: T) {}
//~^ ERROR: destructor of `T` cannot be evaluated at compile-time [E0493]

fn main() {}
