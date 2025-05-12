//@ aux-build:block-on.rs
//@ edition:2021

extern crate block_on;

// Make sure that we can't make an async closure that evaluates to a self-borrow.
// i.e. that the generator may reference captures, but the future's return type can't.

fn main() {
    block_on::block_on(async {
        let s = String::new();
        let x = async move || -> &String { &s };
        //~^ ERROR lifetime may not live long enough

        let s = String::new();
        let x = async move || { &s };
        //~^ ERROR lifetime may not live long enough
    });
}
