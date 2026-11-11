//@ compile-flags: -Zvalidate-mir
//@ edition: 2021

// NOT copy.
struct Ty;

fn hello(x: &Ty) {
    let c = async || {
        *x;
        //~^ ERROR cannot move out of `*x` which is behind a shared reference
    };
}

fn main() {}
