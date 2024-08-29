//@ compile-flags: -Zvalidate-mir
//@ edition: 2021

#![feature(async_closure)]

// NOT copy.
struct Ty;

fn hello(x: &Ty) {
    let c = async || {
        *x;
        //~^ ERROR cannot move out of `*x` which is behind a shared reference
    };
}

fn main() {}
