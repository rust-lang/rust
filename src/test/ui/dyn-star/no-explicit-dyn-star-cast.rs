use std::fmt::Debug;

fn make_dyn_star() {
    let i = 42usize;
    let dyn_i: dyn* Debug = i as dyn* Debug;
    //~^ ERROR casting `usize` as `dyn* Debug` is invalid
    //~| ERROR dyn* trait objects are unstable
    //~| ERROR dyn* trait objects are unstable
}

fn main() {
    make_dyn_star();
}
