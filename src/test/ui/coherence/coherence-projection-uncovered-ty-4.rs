// aux-build:coherence_projection_assoc.rs

extern crate coherence_projection_assoc as lib;
use lib::Foreign;

trait Id {
    type Assoc;
}

impl<T> Id for T {
    type Assoc = T;
}

impl<T> Foreign<(), Vec<T>> for <T as Id>::Assoc {
//~^ ERROR E0210
    type Assoc = usize;
}

fn main() {}
