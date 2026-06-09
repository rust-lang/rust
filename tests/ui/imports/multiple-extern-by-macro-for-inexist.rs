//@ edition: 2021

// issue#128813

extern crate non_existent;
//~^ ERROR: can't find crate for `non_existent`

macro_rules! m {
    () => {
        extern crate std as non_existent;
        //~^ ERROR: the name `non_existent` is defined multiple times
    };
}

m!();

fn main() {
    use ::non_existent;
}
