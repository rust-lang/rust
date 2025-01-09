//@ run-pass
//@ compile-flags: -Zcontract-checks=yes
#![feature(rustc_contracts)]

struct Outer { outer: std::cell::Cell<i32> }

fn outer(x: Outer)
    rustc_contract_requires(|| x.outer.get() > 0)
{
    let inner_closure = || { };
    x.outer.set(0);
    inner_closure();
}

fn main() {
    outer(Outer { outer: 1.into() });
}
