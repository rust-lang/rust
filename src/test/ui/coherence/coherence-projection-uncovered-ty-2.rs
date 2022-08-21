// Here we expect that orphan rule is violated
// because T is an uncovered parameter appearing
// before the first local (Issue #99554)

// aux-build:coherence_projection_assoc.rs

extern crate coherence_projection_assoc as lib;
use lib::Foreign;

trait Id {
    type Assoc;
}

impl<T> Id for T {
    type Assoc = T;
}

pub struct B;
impl<T> Foreign<B, T> for <Vec<Vec<T>> as Id>::Assoc {
    //~^ ERROR E0210
    type Assoc = usize;
}

fn main() {}
