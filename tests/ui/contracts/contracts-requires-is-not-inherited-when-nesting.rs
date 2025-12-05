//@ run-pass
//@ compile-flags: -Zcontract-checks=yes
#![feature(contracts)]
//~^ WARN the feature `contracts` is incomplete and may not be safe to use and/or cause compiler crashes [incomplete_features]

struct Outer { outer: std::cell::Cell<i32> }

#[core::contracts::requires(x.outer.get() > 0)]
fn outer(x: Outer) {
    let inner_closure = || { };
    x.outer.set(0);
    inner_closure();
}

fn main() {
    outer(Outer { outer: 1.into() });
}
