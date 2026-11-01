//@ compile-flags: --crate-type=lib
//@ edition: 2021
#![expect(incomplete_features)]
#![feature(contracts)]

#[core::contracts::ensures(|ret| *ret)]
//~^ ERROR contract annotations are not yet supported on async or gen functions
async fn _always_true(b: bool) -> bool {
    b
}
