// aux-build:m1.rs


extern crate m1;

struct X {
}

impl m1::X for X { //~ ERROR not all trait items implemented
}

fn main() {}
