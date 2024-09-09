//@ edition: 2021

// issue#128813

extern crate core;

macro_rules! m {
    () => {
        extern crate std as core;
        //~^ ERROR: the name `core` is defined multiple times
    };
}

m!();

fn main() {
    use ::core;
}
