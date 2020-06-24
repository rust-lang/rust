// aux-build:first-second.rs
// FIXME: The spans here are bad, see PR #73084

extern crate first_second;
use first_second::*;

macro_rules! produce_it {
    ($name:ident) => {
        #[first] //~ ERROR cannot find type
        struct $name {
            field: MissingType
        }
    }
}

produce_it!(MyName);

fn main() {
    println!("Hello, world!");
}
