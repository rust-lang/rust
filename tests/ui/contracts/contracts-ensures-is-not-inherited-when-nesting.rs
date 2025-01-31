//@ run-pass
//@ compile-flags: -Zcontract-checks=yes
#![feature(contracts)]
//~^ WARN the feature `contracts` is incomplete and may not be safe to use and/or cause compiler crashes [incomplete_features]

#[core::contracts::ensures(|ret| *ret > 0)]
fn outer() -> i32 {
    let inner_closure = || -> i32 { 0 };
    inner_closure();
    10
}

fn main() {
    outer();
}
