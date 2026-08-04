//@ run-rustfix
//@ rustfix-only-machine-applicable

// Verify the #160431 rewrite is MachineApplicable: rustfix applies it and the result compiles
// without `#![feature(closure_lifetime_binder)]`.

fn main() {
    let _cl = for<'a> |x: &'a str| -> (&'a str, &'a str) { x.split_at(0) };
    //~^ ERROR `for<...>` binders for closures are experimental
}
