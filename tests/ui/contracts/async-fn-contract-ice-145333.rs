//@ compile-flags: --crate-type=lib
//@ edition: 2021
#![feature(contracts)]
//~^ WARN the feature `contracts` is incomplete

#[core::contracts::ensures(|ret| *ret)]
//~^ ERROR contract annotations are not yet supported on async or gen functions
async fn _always_true(b: bool) -> bool {
    b
}
