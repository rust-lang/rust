//@ edition:2018

// Regression test for #95307.
// The ICE occurred on all the editions, specifying edition:2018 to reduce diagnostics.

pub trait C {
    async fn new() -> [u8; _];
    //~^ ERROR: using `_` for array lengths is unstable
    //~| ERROR: in expressions, `_` can only be used on the left-hand side of an assignment
}

fn main() {}
