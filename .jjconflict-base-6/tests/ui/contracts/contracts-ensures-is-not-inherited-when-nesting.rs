//@ run-pass
//@ compile-flags: -Zcontract-checks=yes
#![expect(incomplete_features)]
#![feature(contracts)]

#[core::contracts::ensures(|ret| *ret > 0)]
fn outer() -> i32 {
    let inner_closure = || -> i32 { 0 };
    inner_closure();
    10
}

fn main() {
    outer();
}
