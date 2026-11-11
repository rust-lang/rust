//@ edition: 2021

#![deny(closure_returning_async_block)]

fn main() {
    let x = || async {};
    //~^ ERROR closure returning async block can be made into an async closure

    let x = || async move {};
    //~^ ERROR closure returning async block can be made into an async closure

    let x = move || async move {};
    //~^ ERROR closure returning async block can be made into an async closure

    let x = move || async {};
    //~^ ERROR closure returning async block can be made into an async closure

    let x = || {{ async {} }};
    //~^ ERROR closure returning async block can be made into an async closure
}
