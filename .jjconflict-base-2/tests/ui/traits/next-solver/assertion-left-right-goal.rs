//@ compile-flags: -Znext-solver
//@ edition: 2018
//@ check-fail

pub trait Bar: Sized {
    async fn new() -> impl Fn<()> {
        //~^ ERROR: the precise format of `Fn`-family traits' type parameters is subject to change
        //~| ERROR: the precise format of `Fn`-family traits' type parameters is subject to change
        //~| ERROR: the precise format of `Fn`-family traits' type parameters is subject to change
        //~| ERROR: the precise format of `Fn`-family traits' type parameters is subject to change
        //~| ERROR: mismatched types
        async {}
    }
}

fn main() {}
