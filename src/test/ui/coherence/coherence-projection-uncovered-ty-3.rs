// aux-build:coherence_projection_assoc.rs

extern crate coherence_projection_assoc as lib;
use lib::Foreign;

trait Id {
    type Assoc;
}

impl<T> Id for T {
    type Assoc = T;
}

pub struct Local<T>(T);
impl<T> Foreign<<T as Id>::Assoc, Local<T>> for () {
    //~^ ERROR E0210
    type Assoc = usize;
}

fn main() {}
