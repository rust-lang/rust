//@ edition:2018

// Regression test for #95307.
// The ICE occurred on all the editions, specifying edition:2018 to reduce diagnostics.

pub trait C {
    async fn new() -> [u8; _];
    //~^ ERROR: the placeholder `_` is not allowed within types on item signatures for functions
    //~| ERROR using `_` for array lengths is unstable
}

fn main() {}
