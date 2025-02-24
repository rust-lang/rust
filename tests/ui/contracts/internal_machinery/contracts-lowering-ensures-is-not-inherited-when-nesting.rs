//@ run-pass
//@ compile-flags: -Zcontract-checks=yes
#![feature(contracts_internals)]

fn outer() -> i32
    contract_ensures(|ret| *ret > 0)
{
    let inner_closure = || -> i32 { 0 };
    inner_closure();
    10
}

fn main() {
    outer();
}
