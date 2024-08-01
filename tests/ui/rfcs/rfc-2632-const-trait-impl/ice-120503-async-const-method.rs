//@ edition: 2021
#![feature(effects)] //~ WARN the feature `effects` is incomplete

trait MyTrait {}

impl MyTrait for i32 {
    async const fn bar(&self) {
        //~^ ERROR expected one of `extern`
        //~| ERROR functions in trait impls cannot be declared const
        //~| ERROR functions cannot be both `const` and `async`
        //~| ERROR method `bar` is not a member
        main8().await;
        //~^ ERROR cannot find function
    }
}

fn main() {}
