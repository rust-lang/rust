//@ run-pass
//@ compile-flags: -Zcontract-checks=yes
#![expect(incomplete_features)]
#![feature(contracts)]

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
