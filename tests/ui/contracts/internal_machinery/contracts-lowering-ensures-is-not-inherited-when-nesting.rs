//@ run-pass
//@ compile-flags: -Zcontract-checks=yes
#![feature(rustc_contracts_internals)]

fn outer() -> i32
    rustc_contract_ensures(|ret| *ret > 0)
{
    let inner_closure = || -> i32 { 0 };
    inner_closure();
    10
}

fn main() {
    outer();
}
