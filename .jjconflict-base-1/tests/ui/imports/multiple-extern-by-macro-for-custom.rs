//@ edition: 2021
//@ aux-build: empty.rs

// issue#128813

extern crate empty;

macro_rules! m {
    () => {
        extern crate std as empty;
        //~^ ERROR: the name `empty` is defined multiple times
    };
}

m!();

fn main() {
    use ::empty;
}
