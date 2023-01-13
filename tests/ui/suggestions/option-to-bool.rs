#![cfg_attr(let_chains, feature(let_chains))]

fn foo(x: Option<i32>) {
    if true && x {}
    //~^ ERROR mismatched types
    //~| HELP use `Option::is_some` to test if the `Option` has a value
}

fn main() {}
