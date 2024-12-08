//@ proc-macro: first-second.rs

extern crate first_second;
use first_second::*;

macro_rules! produce_it {
    ($name:ident) => {
        #[first]
        struct $name {
            field: MissingType //~ ERROR cannot find type
        }
    }
}

produce_it!(MyName);

fn main() {
    println!("Hello, world!");
}
