//@ revisions: ed2015 ed2021
//@[ed2015] edition: 2015
//@[ed2021] edition: 2021

// issue#128813

extern crate core as _;

macro_rules! m {
    () => {
        extern crate std as _;
    };
}

m!();

fn main() {
    use ::_;
    //~^ ERROR: expected identifier, found reserved identifier `_`
}
